"""Behavior cloning trainer — jointly trains PIP MDN and guidance FF
from ccKill grid telemetry (envelope targets only)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import json
import sqlite3
import numpy as np

from core.players.pipPlayer      import PipPlayer
from core.players.guidancePlayer import GuidancePlayer
from utils.PlotUtils             import PlotUtils as plot

# paths
PYFADS_DIR    = os.path.expanduser("~/Documents/source/pyfads")
ENVELOPE_PATH = os.path.join(PYFADS_DIR, "grid_telemetry/envelope.json")
TELEMETRY_DIR = os.path.join(PYFADS_DIR, "grid_telemetry")


def loadTelemetry(dbPath:str, vid:str, table:str) -> np.ndarray:
    """Load a telemetry variable from a saved .db file."""
    conn = sqlite3.connect(dbPath)
    cur = conn.cursor()
    cur.execute(f"SELECT data FROM {table} WHERE vid=?", (vid,))
    rows = cur.fetchall()
    conn.close()
    if not rows:
        raise ValueError(f"No data for vid={vid} table={table} in {dbPath}")
    arrays = [np.frombuffer(r[0], dtype=np.float64) for r in rows]
    return np.array(arrays)


def loadEnvelope():
    """Load envelope.json and return list of {target, guide, vid}."""
    with open(ENVELOPE_PATH) as f:
        data = json.load(f)
    return data["envelope"]


def buildDataset(envelope:list):
    """Build training arrays from envelope telemetry.

    Returns:
        pipInputs:   (N, 10)  — nav state + target
        pipTargets:  (N, 3)   — ccKill PIP
        gdcInputs:   (N, 24)  — nav state + MDN placeholder (zeros for now)
        gdcTargets:  (N, 3)   — ccKill accCmd
    """
    allPipIn, allPipTgt = [], []
    allGdcIn, allGdcTgt = [], []

    for entry in envelope:
        vid = entry["vid"]
        targetPos = np.array(entry["target"])
        dbPath = os.path.join(TELEMETRY_DIR, f"{vid}.db")

        # load per-timestep telemetry
        posNed  = loadTelemetry(dbPath, vid, "eom_posNed")
        velNed  = loadTelemetry(dbPath, vid, "nav_velNedEst")
        mach    = loadTelemetry(dbPath, vid, "nav_mach")
        pip     = loadTelemetry(dbPath, vid, "gdc_pip")
        accCmd  = loadTelemetry(dbPath, vid, "gdc_accCmdNed")

        nSteps = len(posNed)
        targetBroadcast = np.tile(targetPos, (nSteps, 1))

        # PIP net inputs: posNed(3) + velNed(3) + mach(1) + target(3) = 10
        if mach.ndim == 1:
            mach = mach.reshape(-1, 1)
        pipIn = np.hstack([posNed, velNed, mach, targetBroadcast])
        allPipIn.append(pipIn)
        allPipTgt.append(pip)

        # Guidance net inputs: nav(10) + MDN output(14) = 24
        # During behavior cloning, MDN output is zeros (placeholder);
        # the trainer fills it from the PIP net's forward pass below
        mdnPlaceholder = np.zeros((nSteps, 14))
        gdcIn = np.hstack([pipIn, mdnPlaceholder])
        allGdcIn.append(gdcIn)
        allGdcTgt.append(accCmd)

    return (np.vstack(allPipIn),  np.vstack(allPipTgt),
            np.vstack(allGdcIn),  np.vstack(allGdcTgt))


def train(numEpochs:int=200, batchSize:int=256, lr:float=1e-3):
    envelope = loadEnvelope()
    print(f"Loaded {len(envelope)} envelope targets")

    pipInputs, pipTargets, gdcInputs, gdcTargets = buildDataset(envelope)
    N = len(pipInputs)
    print(f"Dataset: {N} timesteps")

    pipPlayer = PipPlayer(lr=lr)
    gdcPlayer = GuidancePlayer(lr=lr)

    pipLosses = []
    gdcLosses = []

    for epoch in range(numEpochs):
        # shuffle
        idx = np.random.permutation(N)
        pipIn  = pipInputs[idx]
        pipTgt = pipTargets[idx]
        gdcIn  = gdcInputs[idx]
        gdcTgt = gdcTargets[idx]

        epochPipLoss = 0.
        epochGdcLoss = 0.
        nBatches = 0

        for i in range(0, N, batchSize):
            bPipIn  = pipIn[i:i+batchSize]
            bPipTgt = pipTgt[i:i+batchSize]
            bGdcTgt = gdcTgt[i:i+batchSize]

            # train PIP net
            pipLoss = pipPlayer.trainStep(bPipIn, bPipTgt)

            # get PIP net's output to feed guidance net
            means, vars, weights = pipPlayer.agent.predict(bPipIn)
            mdnFlat = np.zeros((len(bPipIn), 14))
            for k in range(2):
                mdnFlat[:, k*7:k*7+3]   = means[:, k]
                mdnFlat[:, k*7+3:k*7+6] = vars[:, k]
                mdnFlat[:, k*7+6]       = weights[:, k]

            # train guidance net with PIP output
            bGdcIn = np.hstack([bPipIn, mdnFlat])
            gdcLoss = gdcPlayer.trainStep(bGdcIn, bGdcTgt)

            epochPipLoss += pipLoss
            epochGdcLoss += gdcLoss
            nBatches += 1

        avgPip = epochPipLoss / nBatches
        avgGdc = epochGdcLoss / nBatches
        pipLosses.append(avgPip)
        gdcLosses.append(avgGdc)

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:>3}: PIP NLL={avgPip:.4f}  "
                  f"GDC MSE={avgGdc:.4f}")

    # save models
    pipPlayer.save("./networks/pipNet.pth")
    gdcPlayer.save("./networks/guidanceNet.pth")
    print(f"\nModels saved to ./networks/")

    # plot learning curves
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(pipLosses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('NLL')
    ax1.set_title('PIP MDN Loss')
    ax1.grid(True)

    ax2.plot(gdcLosses)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.set_title('Guidance Net Loss')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('./networks/training_curves.png', dpi=150)
    print(f"Training curves saved to ./networks/training_curves.png")


if __name__ == '__main__':
    train()
