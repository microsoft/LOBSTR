"""
Copyright (C) 2023 Microsoft Corporation
"""
import os
import random
import json
import io
import string
import re
import argparse

import numpy as np
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdAbbreviations
from rdkit.Geometry.rdGeometry import Point2D

# THIS IS ESSENTIAL TO MATCH FIGURES PRODUCED IN IPYTHON
plt.rcParams['figure.dpi'] = 72

num_bond_classes = 23
num_bond_direction_classes = 7

source_abbrev_defns='''# Translations of superatom labels to SMILES.
# First atom of SMILES string should be the one connected to the rest of
# the molecule.
# Empty lines and lines starting with # are ignored.
# Also check spelling.txt to see that the superatom label
# is correctly spelled.

#Me	 C
MeO      OC
MeS      SC
MeN      NC
CF       CF
CF3      C(F)(F)F
CN       C#N
F3CN     NC(F)(F)F
Ph       c1ccccc1
NO       N=O
NO2      N(=O)=O
N(OH)CH3 N(O)C
SO3H     S(=O)(=O)O
COOH     C(=O)O
nBu      CCCC
EtO      OCC
OiBu     OCC(C)C
iPr      CCC
tBu      C(C)(C)C
Ac       C(=O)C
AcO      OC(=O)C
NHAc     NC(=O)C
#OR       O*
#BzO      OCc1ccccc1
BzO OC(=O)C1=CC=CC=C1
THPO     O[C@@H]1OCCCC1

CHO    C=O
NOH    NO 

# Added  release 1.3.0
CO2Et C(=O)OCC
CO2Me C(=O)OC
MeO2S S(=O)(=O)C
NMe2  N(C)C
#CO2R  C(=O)O*
ZNH   NC(=O)OCC1=CC=CC=C1
HOCH2 CO
H2NCH2 CN
Et CC
BnO OCC1=CC=CC=C1
AmNH NCCCCC
AmO OCCCCC
AmO2C C(=O)OCCCCC
AmS SCCCCC
BnNH NCC1=CC=CC=C1
BnO2C C(=O)OCC1=CC=CC=C1
Bu3Sn [Sn](CCCC)(CCCC)CCCC
BuNH    NCCCC
BuO     OCCCC
BuO2C   C(=O)OCCCC
BuS     SCCCC
CBr3    C(Br)(Br)Br
CbzNH   NC(=O)OCC1=CC=CC=C1
CCl3    C(Cl)(Cl)Cl
ClSO2	S(=O)(=O)Cl
COBr    C(=O)Br
COBu    C(=O)CCCC
COCF3   C(=O)C(F)(F)F
COCl    C(=O)Cl
COCO    C(=O)C=O
COEt    C(=O)CC
COF     C(=O)F
COMe    C(=O)C
OCOMe OC(=O)C
CONH2   C(=O)N
CONHEt  C(=O)NCC
CONHMe  C(=O)NC
COSH    C(=O)S
Et2N    N(CC)CC
Et3N    N(CC)(CC)CC
EtNH    NCC
H2NSO2  S(=O)(N)=O
HONH    ON
Me2N    N(C)C
NCO     N=C=O
NCS     N=C=S
NHAm    NCCCCC
NHBn    NCC1=CC=CC=C1
NHBu    NCCCC
NHEt    NCC
NHOH    NO
NHPr    NCCC
NO      N=O
POEt2   P(OCC)OCC
POEt3   P(OCC)(OCC)OCC
POOEt2  P(=O)(OCC)OCC
PrNH    CCCN
SEt     SCC

BOC C(=O)OC(C)(C)C
MsO OS(=O)(=O)C
OTos OS(=O)(=O)c1ccc(C)cc1
Tos S(=O)(=O)c1ccc(C)cc1
C8H CCCCCCCC
C6H CCCCCC
CH2CH3 CC
N(CH2CH3)2 N(CC)CC
N(CH2CH2CH3)2 N(CCC)CCC
C(CH3)3 C(C)(C)C
COCH3 C(=O)C
CH(CH<sub>3</sub>)2 C(C)C
OCF3 OC(F)(F)F
OCCl3 OC(Cl)(Cl)Cl
OCF2H OC(F)F
SO2Me S(=O)(=O)C
OCH2CO2H OCC(=O)O
OCH2CO2Et OCC(=O)OCC
BOC2N N(C(=O)OC(C)(C)C)C(=O)OC(C)(C)C
BOCHN NC(=O)OC(C)(C)C
NHCbz NC(=O)OCc1ccccc1
OCH2CF3 OCC(F)(F)F
NHSO2BU NS(=O)(=O)CCCC
NHSO2Me NS(=O)(=O)C
MeO2SO OS(=O)(=O)C
NHCOOEt NC(=O)OCC
NHCH3 NC
H4NOOC C(=O)ON
C3H7 CCC
C2H5 CC
NHNH2 NN
OCH2CH2OH OCCO
OCH2CHOHCH2OH OCC(O)CO
OCH2CHOHCH2NH OCC(O)CN
NHNHCOCH3 NNC(=O)C
NHNHCOCF3 NNC(=O)C(F)(F)F
NHCOCF3 NC(=O)C(F)(F)F
CO2CysPr C(=O)ON[C@H](CS)C(=O)CCC
HOH2C CO
H3CHN NC
H3CO2C C(=O)OC
CF3CH2 CC(F)(F)F
OBOC OC(=O)OC(C)(C)C
Bn2N N(Cc1ccccc1)Cc1ccccc1
F5S S(F)(F)(F)(F)F
PPh2 P(c1ccccc1)c1ccccc1
PPh3 P(c1ccccc1)(c1ccccc1)c1ccccc1
OCH2Ph OCc1ccccc1
CH2OMe COC
PMBN NCc1ccc(OC)cc1
SO2 S(=O)=O
NH3Cl NCl
CF2CF3 C(F)(F)C(F)(F)F
CF2CF2H C(F)(F)C(F)(F)
Bn Cc1ccccc1
OCH2Ph OCc1ccccc1
COOCH2Ph C(=O)OCc1ccccc1
Ph3CO OC(c1ccccc1)(c1ccccc1)c1ccccc1
Ph3C C(c1ccccc1)(c1ccccc1)c1ccccc1
Me2NO2S S(C)(C)N(=O)=O
SO3Na S(=O)(=O)(=O)[Na]
OSO2Ph OS(=O)(=O)c1ccccc1
(CH2)5Br CCCCCBr
OPh Oc1ccccc1
SPh Sc1ccccc1
NHPh Nc1ccccc1

CONEt2 C(=O)N(CC)CC
CONMe2 C(=O)N(C)C
EtO2CHN NC(=O)OCC
H4NO3S S(=O)(=O)ON
TMS [Si](C)(C)(C)
COCOOCH2CH3 C(=O)C(=O)OCC
OCH2CN OCC#N
#
#  these are useful for expanding superatoms, but not helpful for collapsing them
# Xx [*]
# X  [*]
# Y [*]
# Z [*]
# R [*]
# R1 [*]
# R2 [*]
# R3 [*]
# R4 [*]
# R5 [*]
# R6 [*]
# R7 [*]
# R8 [*]
# R9 [*]
# R10 [*]
# Y2 [*]
#D [*]
'''
# Make standard abbreviation definitions:
defns = []
for l in source_abbrev_defns.split('\n'):
    for num in range(10):
        l = l.replace(str(num), "<sub>{}</sub>".format(str(num)))
    if not l or l[0]=='#':
        continue
    defn = re.sub(r'[ ]+','\t',l)
    sma = defn.split('\t')[1]
    # use the length of the SMARTS as a crude size sort
    defns.append((len(sma),l))
abbrev_defns = '\n'.join([x[1] for x in sorted(defns,reverse=True)])

# Make wildcard abbreviation definitions:
letters_with_numbers = ['R', 'X', 'Y', 'Z']
other_letters = ['A', 'D']
all_letters = letters_with_numbers + other_letters
script_patterns = ["<sub>{}</sub>", "<sub>{}</sub>", "{}"]
numbers = [str(elem) for elem in range(10)]
defns = []
for line in source_abbrev_defns.split('\n'):
    for letter in all_letters:
        number_patterns = ['']
        if letter in letters_with_numbers:
            number_patterns += numbers
        for number_pattern in number_patterns:
            for script in script_patterns:
                l = line
                if not l or l[0]=='#':
                    continue
                l = re.sub(r'[ ]+','\t',l)
                s = l.split('\t')
                s[0] = letter + script.format(str(number_pattern))
                l = ' '.join(s)
                defn = re.sub(r'[ ]+','\t',l)
                sma = defn.split('\t')[1]
                # use the length of the SMARTS as a crude size sort
                defns.append((len(sma),l))
new_lines = list(set([x[1] for x in sorted(defns,reverse=True)]))
wildcard_abbrev_defns = '\n'.join(new_lines)

abbrevs = rdAbbreviations.ParseAbbreviations(abbrev_defns)
wildcard_abbrevs = list(rdAbbreviations.ParseAbbreviations(wildcard_abbrev_defns))

bond_to_idx = {str(v): k + 1 for k, v in rdkit.Chem.rdchem.BondType.values.items()}
bond_to_idx[None] = 0
idx_to_bond = [None] + list(rdkit.Chem.rdchem.BondType.values.values())
print(len(idx_to_bond))

bond_direction_to_idx = {str(v): k for k, v in rdkit.Chem.BondDir.values.items()}
idx_to_bond_direction = list(rdkit.Chem.BondDir.values.values())

formal_charge_to_idx = {v: v+6 for v in range(-6, 7)}
idx_to_formal_charge = list(range(-6, 7))


random_word = lambda n: ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])
def random_string(min_num_words=1, max_num_words=6, min_word_length=2, max_word_length=10):
    return ' '.join([random_word(random.randint(min_word_length, max_word_length)) for elem in range(random.randint(min_num_words, max_num_words))])


def smiles2image_augmented(smiles, width=800, height=800, return_coords=False, font_size_min=30, font_size_max=70,
                           padding_range=[0.02, 0.15], atom_label_padding_range=[0, 0.3],
                           line_width_min=1, line_width_max=8, atom_noise_range=[0, 0.12],
                           rotate_range=[-90, 90], explicit_carbon_prob=0.2,
                           explicit_methyl_prob=0.15, annotation_probs=[0.5, 0.3, 0.2], atom_index_prob=0.1):
    
    m = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(m)
    m = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(m)
    
    num_atoms = len(m.GetAtoms())
    font_size_min = round(font_size_min / (num_atoms ** 0.25))
    font_size_max = round(font_size_max / (num_atoms ** 0.25))
    line_width_max = round(line_width_max / (num_atoms ** 0.25))
    
    for bond in m.GetBonds():
        if random.random() < 0.1:
            bond.SetBondDir(rdkit.Chem.rdchem.BondDir.UNKNOWN)
    
    font_size = random.randint(font_size_min, font_size_max)
    padding = random.uniform(padding_range[0], padding_range[1])
    atom_label_padding = random.uniform(atom_label_padding_range[0], atom_label_padding_range[1])
    line_width = random.randint(line_width_min, line_width_max)
    atom_noise = random.uniform(atom_noise_range[0], atom_noise_range[1])
    rotate = random.randint(rotate_range[0], rotate_range[1])
    #font = random.choices(["comic.ttf", "times.ttf", "arial.ttf", "ITCKRIST.TTF", "segoepr.ttf"], k=1)[0]
    comic_mode = random.choices([True, False, False], k=1)[0]
    explicit_carbon = random.choices([True, False], weights=[explicit_carbon_prob, 1-explicit_carbon_prob], k=1)[0]
    explicit_methyl = random.choices([True, False], weights=[explicit_methyl_prob, 1-explicit_methyl_prob], k=1)[0]
    
    for idx, atom in enumerate(m.GetAtoms()):
        pos = m.GetConformer().GetAtomPosition(idx)
        pos.x += random.uniform(-atom_noise, atom_noise)
        pos.y += random.uniform(-atom_noise, atom_noise)
        m.GetConformer().SetAtomPosition(idx, pos)    
        
    if explicit_carbon:
        for atom in m.GetAtoms():
            if atom.GetAtomicNum() == 6:
                atom.SetProp("atomLabel", atom.GetSymbol())
                
    if random.random() < atom_index_prob:
        for atom in m.GetAtoms():
            atom.SetProp("atomNote", str(atom.GetIdx()+1))
    
    d2d = rdMolDraw2D.MolDraw2DCairo(width, height)
    d2d.SetLineWidth(line_width)
    d2d.drawOptions().useBWAtomPalette()
    d2d.drawOptions().padding = padding
    d2d.drawOptions().additionalAtomLabelPadding = atom_label_padding
    d2d.drawOptions().rotate = rotate
    d2d.drawOptions().comicMode = comic_mode
    #d2d.drawOptions().fontFile = font
    d2d.drawOptions().minFontSize = font_size
    d2d.drawOptions().maxFontSize = font_size
    d2d.drawOptions().explicitMethyl = explicit_methyl
    d2d.drawOptions().annotationFontScale = 0.7
    m = rdMolDraw2D.PrepareMolForDrawing(m, wavyBonds=True)
    
    if explicit_methyl:
        use_condensed = random.choices([True, False], k=1)[0]
        if use_condensed:
            functional_group = Chem.MolFromSmarts('[CH3]')
            matches = m.GetSubstructMatches(functional_group)
            for match in matches:
                atom_idx = match[0]
                atom = m.GetAtomWithIdx(atom_idx)
                atom.SetProp("atomLabel", 'Me')
    
    d2d.DrawMolecule(m)
    d2d.FinishDrawing()
    image = Image.open(io.BytesIO(d2d.GetDrawingText()))
    
    coords = [list(d2d.GetDrawCoords(idx)) for idx, atom in enumerate(m.GetAtoms())]
    
    # Add random text
    draw = ImageDraw.Draw(image)
    num_strings = random.choices([0, 1, 2], weights=annotation_probs, k=1)[0]
    for string_num in range(num_strings):
        #font_name = random.choices(["comic.ttf", "times.ttf", "arial.ttf", "ITCKRIST.TTF", "segoepr.ttf"], k=1)[0]
        #font = ImageFont.truetype(font_name, random.randint(16, 50))

        good = False
        attempts = 0
        while not good and attempts < 20:
            attempts += 1
            good = True
            
            text_string = random_string(max_num_words=2)
            x = random.uniform(0, width)
            y = random.uniform(0, height)
            #x1, y1, x2, y2 = draw.textbbox([x, y], text_string, font=font)
            x1, y1, x2, y2 = draw.textbbox([x, y], text_string)
            w = x2 - x1
            h = y2 - y1
            text_bbox = [x - w/2 - 20, y - h/2 - 20, x + w/2 + 20, y + h/2 + 20]
            for coord in coords:
                if (coord[0] > text_bbox[0] and coord[0] < text_bbox[2]) and (coord[1] > text_bbox[1] and coord[1] < text_bbox[3]):
                    good = False
                    break

        if good:
            #draw.text((x - w/2, y - h/2), text_string, (0, 0, 0), font=font)
            draw.text((x - w/2, y - h/2), text_string, (0, 0, 0))
    
    node_atomic_nums, node_charges, node_hydrogens, bond_type_matrix, bond_direction_matrix = graph_from_mol(m)
    
    if return_coords:
        return image, node_atomic_nums, node_charges, node_hydrogens, bond_type_matrix, bond_direction_matrix, coords
    
    return image, node_atomic_nums, node_charges, node_hydrogens, bond_type_matrix, bond_direction_matrix


def graph_from_mol(mol):
    node_atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    node_charges = [atom.GetFormalCharge()+6 for atom in mol.GetAtoms()]
    node_hydrogens = [atom.GetTotalNumHs() for atom in mol.GetAtoms()]
    
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    #node_coordinates = [conf.GetAtomPosition(idx) for idx in range(len(node_atomic_nums))]
    #node_coordinates = [[elem[0], elem[1]] for elem in node_coordinates]
    
    # Wedge information (bond direction) is not generated by RDKit by default
    # because RDKit uses a different internal representation for chirality.
    Chem.WedgeMolBonds(mol, conf)

    bond_type_matrix = np.zeros((len(node_atomic_nums), len(node_atomic_nums)), dtype=np.int8)
    bond_direction_matrix = np.zeros((len(node_atomic_nums), len(node_atomic_nums)), dtype=np.int8)
    
    for bond in mol.GetBonds():
        # This is an undirected adjacency matrix: the beginning node does not matter
        bond_type_matrix[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond_to_idx[str(bond.GetBondType())]
        bond_type_matrix[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond_to_idx[str(bond.GetBondType())]
        
        # This is a directed adjacency matrix: the beginning node matters
        bond_direction_matrix[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond_direction_to_idx[str(bond.GetBondDir())]
        
    return node_atomic_nums, node_charges, node_hydrogens, bond_type_matrix, bond_direction_matrix


def mol_from_graph(node_atomic_nums, node_charges, node_hydrogens, node_coordinates, bond_type_matrix,
                  bond_direction_matrix):

    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in np.random.permutation(len(node_atomic_nums)):
        a = Chem.Atom(node_atomic_nums[i])
        a.SetFormalCharge(idx_to_formal_charge[node_charges[i]])
        a.SetNumExplicitHs(node_hydrogens[i])
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx
        
    # Coordinates 
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.Set3D(False)
    for i, (x, y) in enumerate(node_coordinates):
        conf.SetAtomPosition(node_to_idx[i], (x, y, 0))
    mol.AddConformer(conf) 

    # add bonds between adjacent atoms
    for ix in np.random.permutation(len(node_atomic_nums)):
        for iy in np.random.permutation(len(node_atomic_nums)):
            if iy <= ix:
                continue
            bond = bond_type_matrix[ix, iy]

            if bond == 0 or bond == num_bond_classes:
                continue
                
            bond_type = idx_to_bond[int(round(bond))]
            
            if bond_direction_matrix[iy, ix] == 0 or bond_direction_matrix[iy, ix] == num_bond_direction_classes:
                num_bonds = mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
                bond = mol.GetBonds()[num_bonds-1]
                bond.SetBondDir(idx_to_bond_direction[bond_direction_matrix[ix, iy]])
                mol.ReplaceBond(num_bonds-1, bond)
            else:
                num_bonds = mol.AddBond(node_to_idx[iy], node_to_idx[ix], bond_type)
                bond = mol.GetBonds()[num_bonds-1]
                bond.SetBondDir(idx_to_bond_direction[bond_direction_matrix[iy, ix]])
                mol.ReplaceBond(num_bonds-1, bond)
                
    try:
        # Convert RWMol to Mol object
        mol = Chem.Mol(mol)
        mol.UpdatePropertyCache()
        Chem.SanitizeMol(mol)
        Chem.AssignChiralTypesFromBondDirs(mol)
        Chem.WedgeMolBonds(mol, conf)
        Chem.DetectBondStereochemistry(mol)
        Chem.AssignStereochemistry(mol)
        mol = Chem.RemoveHs(mol)
    except:
        return None

    return mol


def draw_wavy_attachment_line(mol, drawer, bond, bi, wavy_color=(0.0, 0.0, 0.0)):
    offset = random.uniform(0.05, 0.1)
    wavy_len = random.uniform(0.7, 0.9)
    n_segments = random.randint(6, 12)    
    
    if bi == bond.GetBeginAtomIdx():
        ei = bond.GetEndAtomIdx()
    else:
        ei = bond.GetBeginAtomIdx()
    bc = drawer.GetDrawCoords(bi)
    ec = drawer.GetDrawCoords(ei)
    bv = bc - ec
    midpoint = bc
    bond_len = ((bc.x - ec.x)**2 + (bc.y - ec.y)**2) ** 0.5
    ba = midpoint + Point2D(bv.y, -bv.x) * (wavy_len * 0.5)
    ea = midpoint + Point2D(-bv.y, bv.x) * (wavy_len * 0.5)
    
    drawer.DrawWavyLine(Point2D(ba.x, ba.y), Point2D(ea.x, ea.y),
                        wavy_color, wavy_color, n_segments, offset*bond_len, rawCoords=True)


def smiles2image_augmented2(smiles, width=800, height=800, return_coords=False, font_size_min=30, font_size_max=70,
                           padding_range=[0.02, 0.15], atom_label_padding_range=[0, 0.3],
                           line_width_min=1, line_width_max=8, atom_noise_range=[0, 0.12],
                           rotate_range=[-90, 90], explicit_carbon_prob=0.15,apply_wildcards_prob=0.33,
                           apply_abbreviations_prob=0.5, annotation_probs=[0.5, 0.3, 0.2], unknown_stereo_prob=0.02,
                           atom_index_prob=0.1, wildcard_attachment_point_prob=0.25):
    
    m = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(m)
    m = Chem.MolFromSmiles(smiles)
    
    num_atoms = len(m.GetAtoms())
    font_size_min = round(font_size_min / (num_atoms ** 0.25))
    font_size_max = round(font_size_max / (num_atoms ** 0.25))
    line_width_max = round(line_width_max / (num_atoms ** 0.25))
    
    wildcard = False
    if random.random() <= apply_wildcards_prob:
        wildcard = True
        random.shuffle(wildcard_abbrevs)
        m = rdAbbreviations.CondenseMolAbbreviations(m, wildcard_abbrevs, maxCoverage=1.0)
    
    #m = Chem.AddHs(m) # TODO: Add diagrams with hydrogens; but have to decide if/how to have the model ignore
    AllChem.Compute2DCoords(m)
    
    for bond in m.GetBonds():
        if random.random() < unknown_stereo_prob:
            bond.SetBondDir(rdkit.Chem.rdchem.BondDir.UNKNOWN)
    
    node_atomic_nums = [atom.GetAtomicNum() for atom in m.GetAtoms()]
    
    font_size = random.randint(font_size_min, font_size_max)
    padding = random.uniform(padding_range[0], padding_range[1])
    atom_label_padding = random.uniform(atom_label_padding_range[0], atom_label_padding_range[1])
    line_width = random.randint(line_width_min, line_width_max)
    atom_noise = random.uniform(atom_noise_range[0], atom_noise_range[1])
    rotate = random.randint(rotate_range[0], rotate_range[1])
    font = random.choices(["comic.ttf", "times.ttf", "arial.ttf", "ITCKRIST.TTF", "segoepr.ttf"], k=1)[0]
    comic_mode = random.choices([True, False, False], k=1)[0]
    explicit_carbon = random.choices([True, False], weights=[explicit_carbon_prob, 1-explicit_carbon_prob], k=1)[0]
    
    d2d = rdMolDraw2D.MolDraw2DCairo(width, height)
    d2d.SetLineWidth(line_width)
    d2d.drawOptions().useBWAtomPalette()
    d2d.drawOptions().padding = padding
    d2d.drawOptions().additionalAtomLabelPadding = atom_label_padding
    d2d.drawOptions().rotate = rotate
    d2d.drawOptions().comicMode = comic_mode
    #d2d.drawOptions().fontFile = font
    d2d.drawOptions().minFontSize = font_size
    d2d.drawOptions().maxFontSize = font_size
    d2d.drawOptions().annotationFontScale = 0.7
    
    m = rdMolDraw2D.PrepareMolForDrawing(m)
    
    selected_APs = []
    for idx, atom in enumerate(m.GetAtoms()):
        if atom.GetAtomicNum() == 0 and random.random() < wildcard_attachment_point_prob:
            bonds = atom.GetBonds()
            if len(bonds) > 1:
                continue
            atom.SetProp("atomLabel", "")
            selected_APs.append(atom)
            
    d2d.DrawMolecule(m)
    
    for atom in selected_APs:
        bonds = atom.GetBonds()
        for bond in bonds:
            draw_wavy_attachment_line(m, d2d, bond, atom.GetIdx())
    
    d2d.FinishDrawing()
    image = Image.open(io.BytesIO(d2d.GetDrawingText()))
    
    atom_notes = []
    for atom in m.GetAtoms():
        atom.SetProp('atomNote',str(atom.GetIdx()))
        atom_notes.append(int(atom.GetProp('atomNote')))
    
    coords = [list(d2d.GetDrawCoords(idx)) for idx, atom in enumerate(m.GetAtoms())]
    
    node_atomic_nums, node_charges, node_hydrogens, bond_type_matrix, bond_direction_matrix = graph_from_mol(m)
    
    if not wildcard and random.random() < apply_abbreviations_prob:
        # Determine which abbreviations are in the molecule and where they occur
        matching_abbrevs = []
        abbrevs_to_apply = set()
        for abb in abbrevs:
            smarts = Chem.MolToSmarts(abb.mol)
            matches = m.GetSubstructMatches(Chem.MolFromSmarts(smarts))
            if len(matches) > 0:
                for match in matches:
                    if len(match) > 0:
                        matching_abbrevs.append([abb, match])
                        abbrevs_to_apply.add(abb)
        # Permute the matching abbreviations to vary which ones get applied when abbreviations intersect
        abbrevs_to_apply = np.random.permutation(list(abbrevs_to_apply)).tolist()

        if len(abbrevs_to_apply) > 0:
            m = rdAbbreviations.CondenseMolAbbreviations(m, abbrevs_to_apply, maxCoverage=1.0)

        abbreviated_atom_notes = [int(atom.GetProp('atomNote')) for atom in m.GetAtoms()]
        # Determine which atoms were removed during abbreviation
        missing_atom_nums = [num for num in atom_notes if not num is None and not num in abbreviated_atom_notes]

        # Determine which abbreviations were actually applied
        applied_abbrevs = []
        for abbrev in matching_abbrevs:
            if len(missing_atom_nums) == 0:
                break
            if len(abbrev[1][2:]) == 0:
                continue
            if set(abbrev[1][2:]).issubset(set(missing_atom_nums)):
                applied_abbrevs.append(abbrev + [abbrev[1][1]])
                missing_atom_nums = [elem for elem in missing_atom_nums if not elem in abbrev[1][2:]]

        # Remove the notes from the atoms so they don't get drawn
        for atom in m.GetAtoms():
            atom.SetProp('atomNote','')
        
#------

        AllChem.Compute2DCoords(m)

        for idx, atom in enumerate(m.GetAtoms()):
            pos = m.GetConformer().GetAtomPosition(idx)
            pos.x += random.uniform(-atom_noise, atom_noise)
            pos.y += random.uniform(-atom_noise, atom_noise)
            m.GetConformer().SetAtomPosition(idx, pos)    

        if explicit_carbon:
            for atom in m.GetAtoms():
                if atom.GetAtomicNum() == 6:
                    atom.SetProp("atomLabel", atom.GetSymbol())
                
        if random.random() < atom_index_prob:
            for atom in m.GetAtoms():
                atom.SetProp("atomNote", str(atom.GetIdx()+1))

        d2d = rdMolDraw2D.MolDraw2DCairo(width, height)
        d2d.SetLineWidth(line_width)
        d2d.drawOptions().useBWAtomPalette()
        d2d.drawOptions().padding = padding
        d2d.drawOptions().additionalAtomLabelPadding = atom_label_padding
        d2d.drawOptions().rotate = rotate
        d2d.drawOptions().comicMode = comic_mode
        #d2d.drawOptions().fontFile = font
        d2d.drawOptions().minFontSize = font_size
        d2d.drawOptions().maxFontSize = font_size
 
        m = rdMolDraw2D.PrepareMolForDrawing(m)

        d2d.DrawMolecule(m)
        
        d2d.FinishDrawing()
        image = Image.open(io.BytesIO(d2d.GetDrawingText()))

        _, _, _, _, abbreviated_bond_direction_matrix = graph_from_mol(m)
        abbreviated_coords = [list(d2d.GetDrawCoords(idx)) for idx, atom in enumerate(m.GetAtoms())]

        # Apply the new coordinates of the atoms to the atoms from the unabbreviated molecule
        for k, v in zip(abbreviated_atom_notes, abbreviated_coords):
            coords[k] = v

        # Collapse the atoms in the unabbreviated molecule to have the same location as the
        # point of abbreviation
        for elem in applied_abbrevs:
            attach_idx = elem[2]
            for modify_idx in elem[1][2:]:
                coords[modify_idx] = coords[attach_idx]

        # Apply the bond directions computed between atoms in the new molecule
        for new_idx1, old_idx1 in enumerate(abbreviated_atom_notes):
            for new_idx2, old_idx2 in enumerate(abbreviated_atom_notes):
                bond_direction_matrix[old_idx1, old_idx2] = abbreviated_bond_direction_matrix[new_idx1, new_idx2]
    
#-----------------------------------------    
    
    # Add random text
    draw = ImageDraw.Draw(image)
    num_strings = random.choices([0, 1, 2], weights=annotation_probs, k=1)[0]
    for string_num in range(num_strings):
        #font_name = random.choices(["comic.ttf", "times.ttf", "arial.ttf", "ITCKRIST.TTF", "segoepr.ttf"], k=1)[0]
        #font = ImageFont.truetype(font_name, random.randint(16, 50))

        good = False
        attempts = 0
        while not good and attempts < 20:
            attempts += 1
            good = True
            
            text_string = random_string(max_num_words=2)
            x = random.uniform(0, width)
            y = random.uniform(0, height)
            #x1, y1, x2, y2 = draw.textbbox([x, y], text_string, font=font)
            x1, y1, x2, y2 = draw.textbbox([x, y], text_string)
            w = x2 - x1
            h = y2 - y1
            text_bbox = [x - w/2 - 20, y - h/2 - 20, x + w/2 + 20, y + h/2 + 20]
            for coord in coords:
                if (coord[0] > text_bbox[0] and coord[0] < text_bbox[2]) and (coord[1] > text_bbox[1] and coord[1] < text_bbox[3]):
                    good = False
                    break

        if good:
            #draw.text((x - w/2, y - h/2), text_string, (0, 0, 0), font=font)
            draw.text((x - w/2, y - h/2), text_string, (0, 0, 0))
    
    if return_coords:
        return image, node_atomic_nums, node_charges, node_hydrogens, bond_type_matrix, bond_direction_matrix, coords
    
    return image, node_atomic_nums, node_charges, node_hydrogens, bond_type_matrix, bond_direction_matrix


def generate_random_skeletal_formula(smiles_string, image_size=(800, 800),
                                     atom_noise_range=(0, 0.1), abbreviation_probability=0.5,
                                     annotation_probs=(0.5, 0.3, 0.2)):

    canon_smiles_string = Chem.CanonSmiles(smiles_string)
    
    try:
        (img, node_atomic_nums, node_charges,
         node_hydrogens, bond_type_matrix,
         bond_direction_matrix, coords) = smiles2image_augmented2(canon_smiles_string, return_coords=True,
                                                                  atom_noise_range=atom_noise_range,
                                                                  annotation_probs=annotation_probs,
                                                                  apply_abbreviations_prob=abbreviation_probability)
        new_mol = mol_from_graph(node_atomic_nums, node_charges, node_hydrogens,
                                 [[x, -y] for x,y in coords], bond_type_matrix,
                                 bond_direction_matrix)
    
        new_smiles_string = Chem.CanonSmiles(Chem.MolToSmiles(new_mol))
        
        if '*' in new_smiles_string:
            canon_smiles_string = new_smiles_string
    
        # This is a hack until I can figure out why the above sometimes fails.
        # There could be a bug in the RDKit abbreviation code that fails to maintain
        # chilarity for the abbreviated atoms.
        if not new_smiles_string == canon_smiles_string :
            raise Exception
    except:
        canon_smiles_string = Chem.CanonSmiles(smiles_string)
        (img, node_atomic_nums, node_charges,
         node_hydrogens, bond_type_matrix,
         bond_direction_matrix, coords) = smiles2image_augmented(canon_smiles_string, return_coords=True,
                                                                 atom_noise_range=atom_noise_range,
                                                                 annotation_probs=annotation_probs)
    
    labels = node_atomic_nums
    
    w, h = img.size
    
    keep_coords = []
    keep_indices = []
    for node_idx, pos in enumerate(coords):
        if pos[0] < 0 or pos[0] > w or pos[1] < 0 or pos[1] > h:
            continue

        keep_coords.append(pos)
        keep_indices.append(node_idx)

    labels = [labels[node_idx] for node_idx in keep_indices]
    hydrogens_labels = [node_hydrogens[node_idx] for node_idx in keep_indices]
    formal_charge_labels = [node_charges[node_idx] for node_idx in keep_indices]

    # Create target
    target = {}
    target["smiles"] = canon_smiles_string
    target["coords"] = coords
    target["labels"] = labels
    target["hydrogens"] = hydrogens_labels
    target["formal_charges"] = formal_charge_labels
    target["bond_types"] = bond_type_matrix.tolist()
    target["bond_directions"] = bond_direction_matrix.tolist()

    return img, target

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--out_subdir', required=True)
    parser.add_argument('--min_idx', type=int, required=True)
    parser.add_argument('--max_idx', type=int, required=True)
    parser.add_argument('--smiles_filepath', required=True)

    return parser.parse_args()

def main():
    args = {k: v for k, v in get_args().__dict__.items() if not v is None}
    print(args)
    
    parent_directory = args['out_dir']
    restart_baseline = args['min_idx']
    max_number = args['max_idx']
    out_subdir = args['out_subdir']
    smiles_filepath = args['smiles_filepath']

    with open(smiles_filepath, 'r') as f:
        smiles_strings = [elem.strip() for elem in f.readlines()]

    max_number = min(max_number, len(smiles_strings))

    subdirs = ['train', 'test', 'val']
    for subdir in subdirs:
        fullpath = os.path.join(parent_directory, subdir)
        if not os.path.exists(fullpath):
            os.makedirs(fullpath)
            
    output_directory = os.path.join(parent_directory, out_subdir)
    
    files = [file for file in os.listdir(output_directory) if file.endswith(".json")]
    files = [elem for elem in files if int(elem.split('.')[0]) >= restart_baseline and int(elem.split('.')[0]) < max_number]
    jpg_files = [file for file in os.listdir(output_directory) if file.endswith(".jpg")]
    jpg_files = [elem for elem in jpg_files if int(elem.split('.')[0]) >= restart_baseline and int(elem.split('.')[0]) < max_number]
    print(len(files))
    print(len(jpg_files))

    current_nums = set([int(elem.split('.')[0]) for elem in jpg_files])
    new_nums = sorted([elem for elem in range(restart_baseline, max_number) if not elem in current_nums])

    for x in new_nums:
        print("{}             ".format(x), end='\r')

        smiles_string = smiles_strings[x]
        fig_img, labels = generate_random_skeletal_formula(smiles_string) # img_width=1200, img_height=1200

        fig_img = fig_img.convert('RGB')
        fig_img.save('{}/{}.jpg'.format(output_directory, x))

        with open('{}/{}.json'.format(output_directory, x), 'w') as outfile:
            json.dump(labels, outfile)

        del fig_img
        del labels

if __name__ == "__main__":
    main()
