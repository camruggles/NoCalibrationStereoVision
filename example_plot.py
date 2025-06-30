import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

def plot_pts_correspondence(pts1, pts2):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=False, sharey=False)

    # scatter the two sets
    ax1.scatter(*zip(*pts1), s=60, color="tab:blue", label="List 1")
    ax2.scatter(*zip(*pts2), s=60, color="tab:orange", label="List 2")

    ax1.set_title("Points from list 1")
    ax2.set_title("Points from list 2")
    ax1.grid(True); ax2.grid(True)

    # draw correspondence lines (same index → same line)
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        # ConnectionPatch takes care of the coordinate transforms
        con = ConnectionPatch(
            xyA=(x1, y1),   coordsA=ax1.transData,  axesA=ax1,
            xyB=(x2, y2),   coordsB=ax2.transData,  axesB=ax2,
            color="gray", linewidth=1, linestyle="--", alpha=0.6
        )
        fig.add_artist(con)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- example inputs (replace with your own data) ---
    pts1 = [(i, i**0.5) for i in range(12)]           # 12 points
    pts2 = [(i + 0.5, 6 - 0.4*i) for i in range(12)]   # 12 points
    # ---------------------------------------------------
    plot_pts_correspondence(pts1, pts2)