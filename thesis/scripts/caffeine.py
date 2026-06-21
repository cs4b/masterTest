from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
import selfies
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import textwrap
from PIL import Image
import io
import os

# Exact representations for caffeine
smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
mol = Chem.MolFromSmiles(smiles)
rdDepictor.Compute2DCoords(mol)
selfies_str = selfies.encoder(smiles)

# Create RDKit 2D structure image
img = Draw.MolToImage(mol, size=(900, 700))
buf = io.BytesIO()
img.save(buf, format="PNG")
buf.seek(0)
struct_img = Image.open(buf)

# Prepare 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=200)
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# Panel A: 2D structure
axes[0, 0].imshow(struct_img)
axes[0, 0].set_title("A. 2D Structure", fontsize=16, pad=10)

# Panel B: SMILES
axes[0, 1].set_title("B. SMILES", fontsize=16, pad=10)
axes[0, 1].text(
    0.5, 0.5,
    "\n".join(textwrap.wrap(smiles, width=20)),
    ha="center", va="center", fontsize=18, family="monospace"
)

# Panel C: SELFIES
axes[1, 0].set_title("C. SELFIES", fontsize=16, pad=10)
axes[1, 0].text(
    0.5, 0.5,
    "\n".join(textwrap.wrap(selfies_str, width=38)),
    ha="center", va="center", fontsize=13, family="monospace"
)

# Panel D: Molecular graph
axes[1, 1].set_title("D. Molecular Graph", fontsize=16, pad=10)

conf = mol.GetConformer()
coords = np.array([(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y) for i in range(mol.GetNumAtoms())])

# Normalize coordinates to [0,1] and flip y for display
x = coords[:, 0]
y = -coords[:, 1]
x = (x - x.min()) / (x.max() - x.min())
y = (y - y.min()) / (y.max() - y.min())

# Draw bonds
for bond in mol.GetBonds():
    i = bond.GetBeginAtomIdx()
    j = bond.GetEndAtomIdx()
    x1, y1 = x[i], y[i]
    x2, y2 = x[j], y[j]
    order = int(bond.GetBondTypeAsDouble())
    if order == 1:
        axes[1, 1].plot([x1, x2], [y1, y2], linewidth=2, color="black", zorder=1)
    elif order == 2:
        dx, dy = x2 - x1, y2 - y1
        norm = (dx**2 + dy**2) ** 0.5
        ox, oy = -dy / norm * 0.01, dx / norm * 0.01
        axes[1, 1].plot([x1 + ox, x2 + ox], [y1 + oy, y2 + oy], linewidth=2, color="black", zorder=1)
        axes[1, 1].plot([x1 - ox, x2 - ox], [y1 - oy, y2 - oy], linewidth=2, color="black", zorder=1)
    else:
        axes[1, 1].plot([x1, x2], [y1, y2], linewidth=2, color="black", zorder=1)

# Draw atoms
color_map = {"C": "#222222", "N": "#2b6cb0", "O": "#c53030"}
for i, atom in enumerate(mol.GetAtoms()):
    xi, yi = x[i], y[i]
    symbol = atom.GetSymbol()
    color = color_map.get(symbol, "#444444")
    circ = Circle((xi, yi), 0.045, facecolor=color, edgecolor="white", linewidth=1.5, zorder=2)
    axes[1, 1].add_patch(circ)
    axes[1, 1].text(xi, yi, symbol, color="white", fontsize=11, ha="center", va="center", weight="bold", zorder=3)

axes[1, 1].set_xlim(-0.05, 1.05)
axes[1, 1].set_ylim(-0.05, 1.05)
axes[1, 1].set_aspect("equal")

plt.tight_layout()

png_path = "/mnt/data/caffeine_representations_2x2.png"
pdf_path = "/mnt/data/caffeine_representations_2x2.pdf"
plt.savefig(png_path, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
