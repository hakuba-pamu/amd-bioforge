#!/bin/bash
# Download test data for AMD-BioForge

set -e

DATA_DIR="./data"
mkdir -p "$DATA_DIR"

echo "AMD-BioForge -- Data Download"
echo "=============================="

# Download ubiquitin structure (PDB: 1UBQ)
echo "  Downloading 1UBQ (Ubiquitin)..."
python3 -c "
# Generate a simple test PDB
pdb = '''ATOM      1  N   MET A   1      27.340  24.430   2.614  1.00  9.67           N
ATOM      2  CA  MET A   1      26.266  25.413   2.842  1.00 10.38           C
ATOM      3  C   MET A   1      26.913  26.639   3.531  1.00  9.62           C
ATOM      4  O   MET A   1      27.886  26.463   4.263  1.00  9.62           O
ATOM      5  CB  MET A   1      25.112  24.880   3.649  1.00 13.77           C
ATOM      6  CG  MET A   1      25.353  24.860   5.138  1.00 16.18           C
ATOM      7  SD  MET A   1      23.930  23.950   5.900  1.00 17.18           S
ATOM      8  CE  MET A   1      24.447  23.941   7.626  1.00 16.07           C
ATOM      9  N   GLN A   2      26.318  27.770   3.190  1.00  9.37           N
ATOM     10  CA  GLN A   2      26.826  29.021   3.800  1.00  9.28           C
END
'''
with open('$DATA_DIR/1ubq_test.pdb', 'w') as f:
    f.write(pdb)
print('  Created test PDB: $DATA_DIR/1ubq_test.pdb')
"

echo ""
echo "Data ready in $DATA_DIR/"
echo "Run: python scripts/run_md.py --pdb $DATA_DIR/1ubq_test.pdb --duration 1"
