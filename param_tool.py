import os
import sys
import time
import re
from collections import Counter
import numpy as np
try:
    import psiresp
except ModuleNotFoundError:
    psiresp = None  # Или оставить как `pass` для игнорирования
    print("Модуль 'psiresp' не найден. Некоторые функции могут быть недоступны.")

import itertools
import threading
import subprocess
import MDAnalysis as mda

from decimal import Decimal as D

from rdkit.Chem.Draw import IPythonConsole
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS
from typing import Callable


## Вспомогательные мини функции
def get_name(path: str, index=0):
    """
    Аргумент: 
        path - строка с путем к файлу
    Возвращает:
        Имя для ключа в словаре 
    """
    # Сохраняем текущий stderr
    original_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, 'w')
        mol = Chem.MolFromSmiles(path)
        if mol:
            return index
        else:
            name_of_file = os.path.basename(path).split('.')[0]
            return name_of_file
    except Exception as e:
        print(f"Ошибка при обработке '{path}': {e}")
        return None
    finally:
        # Возвращаем stderr на исходное место
        sys.stderr = original_stderr
        
        
## Дектораторы
def data_to_dict(funсtion: Callable):
    '''
    Декоратор для обработки данных различного типа (строка, список, словарь) перед вызовом функции.
    '''
    def wrapper(data: str or list[str] or dict[str, str], **kwarg):
        if isinstance(data, str):
            return {get_name(data): funсtion(data, **kwarg)}
        elif isinstance(data, list):
            return {get_name(str_data, i):funсtion(str_data, **kwarg) for i, str_data in enumerate(data)}
        elif isinstance(data, dict):
            return {legend: funсtion(str_data, **kwarg) for legend, str_data in data.items()}
        else:
            print('Некорректный тип данных ввода', file=sys.stderr)
            return None
    return wrapper

def do_fun_for_2d_list(function: Callable):
    '''
    Декоратор для применения функции к каждому элементу двумерного списка (матрицы).
    '''
    def wrapper(lst: list):
        result = []
        for element in lst:
            if isinstance(element, list):
                # Рекурсивно применяем декоратор к вложенному списку
                result.append(wrapper(element))
            else:
                # Применяем функцию к элементу
                result.append(function(element))
        return result
    return wrapper


## 

@data_to_dict
def read_file(path):
    '''
    Читает первую строку из файла и возвращает её.
    '''
    with open (path, 'r') as f:
        return f.readline().strip()

@data_to_dict
def smi_to_chem(str_smi: str, sanitize=True, addH=True, make_N_root=False):
    """
    Преобразует SMILES-строку в объект молекулы RDKit и добавляет атомы водорода.
    """
    rdkit_mol = Chem.MolFromSmiles(str_smi, sanitize) # переводим во внутренний формат chem
    if addH:
        rdkit_mol = Chem.AddHs(rdkit_mol) # протонируем
        
    # Перенумерация атомов, деля аминогруппу началом молекулы
    if make_N_root:
        # Находим индекс азота из аминогруппы
        root_idx = find_amino_nitrogen(rdkit_mol)
        if root_idx != -1:
            # Получаем SMILES с корневым атомом
            smiles_string = Chem.MolToSmiles(rdkit_mol, canonical=True, allHsExplicit=True, rootedAtAtom=root_idx)
            # Создаем новую молекулу из SMILES
            rdkit_mol = Chem.MolFromSmiles(smiles_string, sanitize=False)
            if rdkit_mol is None:
                raise ValueError("Ошибка при создании молекулы после перенумерации атомов")
        else:
            print("Азот аминогруппы не найден. Корень не изменён.")
        # Chem.SanitizeMol(rdkit_mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE, catchErrors=True)
        Chem.SanitizeMol(rdkit_mol)
        
    # Сортируем атомы, чтобы тяжелые атомы шли первыми, а водороды позже
        atom_order = [atom.GetIdx() for atom in rdkit_mol.GetAtoms() if atom.GetSymbol() != 'H']  # тяжелые атомы
        atom_order += [atom.GetIdx() for atom in rdkit_mol.GetAtoms() if atom.GetSymbol() == 'H']  # водороды
    
    # Перенумерация атомов согласно новому порядку
        new_rdkit_mol = Chem.RenumberAtoms(rdkit_mol, atom_order)
        # Пересчитываем валентности
        return new_rdkit_mol
    else:
        return rdkit_mol
    # if make_N_root:
    #     root_idx = find_amino_nitrogen(rdkit_mol)
    #     if root_idx != -1:
        #     smiles_string = Chem.MolToSmiles(rdkit_mol, canonical=True, 
        #                                      allHsExplicit=True, rootedAtAtom=root_idx)
        #     rdkit_mol = Chem.MolFromSmiles(smiles_string, sanitize=True)
        # return rdkit_mol

# @data_to_dict
# def pdb_to_chem(path_to_pdb, removeHs=False):
#     """
#     Открывает pdb файл при помощи RDKit и преобразует его в двухмерную молекулу для удобного отображения
    
#     """
#     rdkit_mol = Chem.MolFromPDBFile(path_to_pdb, removeHs)
#     # Преобразование трехмерных координат в двумерные
#     AllChem.Compute2DCoords(rdkit_mol)

#     return rdkit_mol

@data_to_dict
def pdb_to_chem(path_to_pdb, removeHs=False, make_N_root=False):
    """
    Открывает PDB файл при помощи RDKit, преобразует его в двухмерную молекулу 
    и сохраняет имена атомов как свойства RDKit-атомов.
    
    Аргументы:
        path_to_pdb (str): Путь к файлу PDB.
        removeHs (bool): Удалять ли гидрогены. По умолчанию False.
        make_N_root (bool): Делать ли атом азота корневым для молекулы. По умолчанию False.
        
    Возвращает:
        Chem.Mol: Молекула RDKit с сохраненными именами атомов.
    """
    # Загрузка молекулы из PDB
    rdkit_mol = Chem.MolFromPDBFile(path_to_pdb, removeHs)
    if not rdkit_mol:
        raise ValueError(f"Не удалось загрузить PDB файл: {path_to_pdb}")

    # Открываем PDB файл для извлечения имен атомов
    with open(path_to_pdb, "r") as pdb_file:
        pdb_lines = pdb_file.readlines()

    atom_names = []
    for line in pdb_lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_name = line.split()[2]  # Имя атома в формате PDB
            atom_names.append(atom_name)

    # Проверка соответствия числа атомов
    if len(atom_names) != rdkit_mol.GetNumAtoms():
        raise ValueError("Число атомов в PDB файле и RDKit молекуле не совпадает!")

    # Присваиваем имена атомов
    for atom, name in zip(rdkit_mol.GetAtoms(), atom_names):
        atom.SetProp("_Name", name)

    # Преобразование трехмерных координат в двумерные
    AllChem.Compute2DCoords(rdkit_mol)

    return rdkit_mol

def draw_mol_with_atom_index(mol, charge_list = None, size=(600,400)):
    """
    Добавляет номера атомов в атрибуты атомов молекулы.
    size=(600,400)
    """
    if charge_list is not None and len(charge_list) == len(mol.GetAtoms()):
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(atom.GetIdx())
            atom.SetDoubleProp('PartialCharge', charge_list[i])
            atom.SetProp('atomNote',str("%4.4f" % charge_list[i]))
    else:   
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(atom.GetIdx())
        
    return Draw.MolToImage(mol, size)   


def modify_atoms_for_charge(mol):
    """
    Модифицирует атомные номера на основе заряда.
    Формальный заряд добавляется к атомному номеру.
    """
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        atom.SetIntProp("OriginalAtomicNum", atom.GetAtomicNum())  # Сохраняем оригинальный номер
        atom.SetAtomicNum(atom.GetAtomicNum() + charge * 100)  # Смещаем атомный номер

def restore_original_atomic_numbers(mol):
    """
    Восстанавливает исходные атомные номера после модификации.
    """
    for atom in mol.GetAtoms():
        if atom.HasProp("OriginalAtomicNum"):
            atom.SetAtomicNum(atom.GetIntProp("OriginalAtomicNum"))
            atom.ClearProp("OriginalAtomicNum")

def match_chem (mol_chem_1, mol_chem_2, compare_any_bond = False ):
    """
    Принимает на вход мономер и полимер (RDKit.Chem) и соотносит их между собой
    Возвращает:
        substructure: общая подструктура мономера и полимера
        dict_mon_pol_matches: словарь сопоставления номеров атомов подструктуры мономера и полимера
    """
    # modify_atoms_for_charge(mol_chem_1)
    # modify_atoms_for_charge(mol_chem_2)

    if compare_any_bond:
        bondCompare = rdFMCS.BondCompare.CompareAny
    else:
        bondCompare = rdFMCS.BondCompare.CompareOrder

    # Используем встроенный механизм для сравнения атомов через свойства
    res = rdFMCS.FindMCS(
                        (mol_chem_1, mol_chem_2),
                        bondCompare=bondCompare,
                        atomCompare=rdFMCS.AtomCompare.CompareElements,  # Сравнение атомов по номерам
                    )
    # restore_original_atomic_numbers(mol_chem_1)
    # restore_original_atomic_numbers(mol_chem_2)
    
    mcs_smarts = res.smartsString # переводим графф в смартс
    substructure = Chem.MolFromSmarts(mcs_smarts) # на основе смартс строим молекулу общей подструктуры 
    # print(mol_chem_2.GetSubstructMatch(res.queryMol))
    
    index_monomer_matches = mol_chem_1.GetSubstructMatch(substructure)
    index_polymer_matches = mol_chem_2.GetSubstructMatch(substructure)
    dict_mon_pol_matches = dict(zip(index_monomer_matches, index_polymer_matches))
    # создаем копию substructure для работы с зарядами
    # substructure_copy = copy_molecule_with_charges(substructure)  
    for pol_atom in mol_chem_2.GetAtoms(): # добавляем заряды на атомы в substructure
        if pol_atom.GetFormalCharge() != 0 and pol_atom.GetIdx() in index_polymer_matches:
            chargAtomIndex = index_polymer_matches.index(pol_atom.GetIdx())
            chargeAtom = pol_atom.GetFormalCharge()
            substructure.GetAtoms()[chargAtomIndex].SetFormalCharge(chargeAtom)
        
    return substructure, dict_mon_pol_matches

def match_mon_to_pol(monomer_dict, polymer_dict, compare_any_bond = False):
    """
    
    Аргументы:
        monomer_dict - словарь мономера (состоит из одной пары имя: rdkit.Chem)
        polymer_dict - словарь полимера (состоит из одной пар имя: rdkit.Chem)
    Возвращает:
        Словарь match_data_dict со следующими подсловарями:
        - substructure - сожердит общие подструктуры мономера и полимера, для каждого из полимеров
        - mon_pol_matches - содержит словари соответствия номеров атомов в общей подструктуре 
        с номерами атомов в полимере 
    """
    try:
        match_data_dict = {'substructure': {}, 
                           'mon_pol_matches': {},
                           'N_match_atoms': {}}
        mon_key, mon_val = next(iter(monomer_dict.items()))

        for pol_key, pol_val in polymer_dict.items():
            sub, dict_mon_pol_matches = match_chem(mon_val, pol_val, compare_any_bond)
            
            match_data_dict['substructure'][pol_key] = sub
            match_data_dict['N_match_atoms'][sub.GetNumAtoms()] = pol_key
            match_data_dict['mon_pol_matches'][pol_key] = dict_mon_pol_matches
            
        return match_data_dict
    except Exception as e:
        print_red(f'Что-то пошло не так!\n{e}')


def draw_mon_pol_match(monomer_chem_dict, polymer_chem_dict, 
                       match_data = False,
                       add_atom_index = False, add_small_atom_index = False, show_any_monomer_matches = False, 
                       n_row = 0, useSVG = True, img_size = (500,300)):
    """
    Отображает сопоставление мономеров и полимеров в виде сетки изображений молекул.
    
    Аргументы:
        monomer_chem_dict (dict): Словарь мономеров.
        polymer_chem_dict (dict): Словарь полимеров.
        match_data (dict): Словарь данных сопоставления мономера с полимерами. По умолчанию False.
        add_atom_index (bool, optional): Добавлять ли номера атомов. По умолчанию False.
        add_small_atom_index (bool, optional): Использовать ли маленькие номера атомов. По умолчанию False.
        show_any_monomer_matches (bool, optional): Показать мономер с подструктурой для каждого полимера. По умолчанию False.
        n_row (int, optional): Количество молекул в строке. По умолчанию = кол-ву полимеров
        useSVG (bool, optional): Использовать ли SVG для изображения. По умолчанию True.
        img_size (tuple, optional): Размер изображения. По умолчанию (500, 300).
    
    Возвращает:
        Drawing: Изображение молекул в виде сетки.
    """
    n_monomers, n_polymers = 1, len(polymer_chem_dict)
    monomer_legend_list, monomer_chem_list = list(monomer_chem_dict.keys()), list(monomer_chem_dict.values())
    polymer_legend_list, polymer_chem_list = list(polymer_chem_dict.keys()), list(polymer_chem_dict.values())
    
    # Если хотим отрисовать подструктуры мономер в составе полимера 
    if match_data:
        monomer_chem_list *= len(match_data['mon_pol_matches'])
        monomer_legend_list *= len(match_data['mon_pol_matches'])
        match_mon_pol = list(match_data['mon_pol_matches'].values())
        monomer_match_list = [list(match_dict.keys()) for match_dict in match_mon_pol]
        polymer_match_list = [list(match_dict.values()) for match_dict in match_mon_pol]
        match_matrix = monomer_match_list + polymer_match_list
    else: # отрисовываем один мономер без сопоставления
        match_matrix = False
        smi_pattern = smi_to_chem('')# Редактируем первую строку в зависимости от кол-ва полимеров
        while n_monomers < n_polymers:
            dif = n_polymers - n_monomers
            if dif % 2 == 0:
                monomer_chem_list.insert(0,smi_pattern)
                monomer_legend_list.insert(0,'')
            else:
                monomer_chem_list.append(smi_pattern)
                monomer_legend_list.append('')
            n_monomers += 1
        
    # Делаем матрицу Chem
    chem_matrix = monomer_chem_list + polymer_chem_list

    # Делаем матрицу легенды
    legend_matrix = monomer_legend_list + polymer_legend_list

    if add_atom_index: 
        for mol in chem_matrix:
            draw_mol_with_atom_index(mol)
    if n_row == 0:
        n_row = n_polymers
        
    IPythonConsole.drawOptions.addAtomIndices = add_small_atom_index
    drawing = Draw.MolsToGridImage(mols = chem_matrix, 
                                   legends = legend_matrix,
                                   highlightAtomLists = match_matrix ,
                                   molsPerRow = n_row,
                                   useSVG=useSVG, 
                                   subImgSize=img_size
                                  )
    IPythonConsole.drawOptions.addAtomIndices = False
    return drawing

def path_parser(path: str, file_types: list):
    """
    Парсит путь и возвращает путь к директории и имя файла без расширений из списка file_types.
    """
    # Форматирование расширений, добавление '.' если отсутствует
    formated_file_types = ['.' + t if not t.startswith('.') else t for t in file_types]
    
    # Разделение пути и имени файла
    path_to_file, file_name = os.path.split(path)
    if path_to_file:
        os.makedirs(path_to_file, exist_ok=True)
    # else:
    #     path_to_file = 'current directory'

    # Удаление указанных расширений из имени файла
    for type_name in formated_file_types:
        file_name = file_name.replace(type_name, '')

    return path_to_file, file_name
        
def find_amino_nitrogen(mol):
    """
    Находит индекс атома азота в составе аминокислоты.
    Предполагается, что любая а.к. имеет вид основу [H][N]CC=O.
    """
    aa_base = '[H][N]CC=O'
    sub_base = mol.GetSubstructMatch(Chem.MolFromSmarts(aa_base))
    if sub_base:
        print(f"atom number: {sub_base[1]} became the root atom: 0")
        return sub_base[1]
    return -1

def save_chem_to_smiles(rdkit_mol, path, canonical=True, allHsExplicit=True, rootedAtAtom=-1):
    path_dir = path.rsplit('/', 1)
    if len(path_dir) > 1:
        os.makedirs(path_dir[0], exist_ok=True) 
    smiles_string = Chem.MolToSmiles(rdkit_mol, canonical = canonical, 
                                     allHsExplicit = allHsExplicit, 
                                     rootedAtAtom = rootedAtAtom)
    with open(f"{path}.smiles", "w") as file:
        file.write(smiles_string)

def save_aa_chem_to_smiles(rdkit_mol, path, canonical=True, allHsExplicit=True, make_N_root=False):
    """
    Сохраняет молекулу в формате SMILES.
    Аргументы:
        rdkit_mol - RDKit.Chem молекула
        path - строка с путем к файлу и его именем для сохранения
        make_N_root - искать ли аминогруппу и делать ли ее началом молекулы
        canonical, allHsExplicit - параметры для генерации SMILES
    """
    if make_N_root:
        rootedAtAtom = find_amino_nitrogen(rdkit_mol)
    else:
        rootedAtAtom = -1

    # Получение пути и имени файла
    path_to_file, file_name = path_parser(path, ['smi', 'smiles'])
    
    # Генерация строки SMILES
    smiles_string = Chem.MolToSmiles(rdkit_mol, canonical=canonical, 
                                     allHsExplicit=allHsExplicit, 
                                     rootedAtAtom=rootedAtAtom)

    # Сохранение в файл
    output_path = f"{path_to_file}/{file_name}" if path_to_file else {file_name}
    with open(f"{output_path}.smiles", "w") as file:
        file.write(smiles_string)
    print(f'{file_name}.smiles saved to {path_to_file or "current directory"}')
        
def save_chem_to_mol(rdkit_mol, path, rootedAtAtom=-1):
    """
    Сохраняет молекулу в формате MOL, устанавливая корневой атом, если задан.
    """
    if rootedAtAtom != -1:
        # Используем SMILES для переупорядочивания атомов
        smiles_string = Chem.MolToSmiles(rdkit_mol, canonical=True, 
                                         allHsExplicit=True, rootedAtAtom=rootedAtAtom)
        rdkit_mol = Chem.MolFromSmiles(smiles_string, sanitize=True)
    
    path_dir = path.rsplit('/', 1)
    if len(path_dir) > 1:
        os.makedirs(path_dir[0], exist_ok=True) 
    with open(f"{path}.mol", "w") as file:
        file.write(Chem.MolToMolBlock(rdkit_mol))
        
def save_chem_to_pdb(rdkit_mol, path, rootedAtAtom=-1):
    """
    Сохраняет молекулу в формате pdb, устанавливая корневой атом, если задан.
    """
    if rootedAtAtom != -1:
        # Используем SMILES для переупорядочивания атомов
        smiles_string = Chem.MolToSmiles(rdkit_mol, canonical=True, 
                                         allHsExplicit=True, rootedAtAtom=rootedAtAtom)
        rdkit_mol = Chem.MolFromSmiles(smiles_string, sanitize=False)
    try:
        Chem.SanitizeMol(rdkit_mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE, catchErrors=True)
    except ValueError as e:
        print("Ошибка санации:", e)
    path_dir = path.rsplit('/', 1)
    if len(path_dir) > 1:
        os.makedirs(path_dir[0], exist_ok=True) 
    AllChem.EmbedMolecule(rdkit_mol)
    AllChem.UFFOptimizeMolecule(rdkit_mol)
    with open(f"{path}.pdb", "w") as file:
        file.write(Chem.MolToPDBBlock(rdkit_mol))
        
def generate_atom_names_by_ref_aa(mod_aa_mol, ref_aa_mol, dict_match, output_path):
    """
    Переименовывает атомы в модифицированной аминокислоте на основе референсной.
    
    Аргументы:
        mod_aa_mol: RDKit объект Chem.Mol для модифицированной молекулы.
        ref_aa_mol: RDKit объект Chem.Mol для референсной молекулы.
        dict_match: Словарь сопоставления индексов атомов (модифицированные -> референсные).
        output_path: Путь для сохранения модифицированного PDB файла.
    """
    # Получаем текущие имена атомов модифицированной молекулы
    all_mod_atom_names = [str(atom.GetProp("_Name")) if atom.HasProp("_Name") else "" for atom in mod_aa_mol.GetAtoms()]
    all_ref_atom_names = [str(atom.GetProp("_Name")) if atom.HasProp("_Name") else "" for atom in ref_aa_mol.GetAtoms()]
    print(all_mod_atom_names)

    for mod_indx, ref_indx in dict_match.items():
        mod_atom_name = all_mod_atom_names[mod_indx]
        ref_atom_name = all_ref_atom_names[ref_indx]

        # Если имя из референсной молекулы уже есть в модифицированной
        if ref_atom_name in all_mod_atom_names:
            new_name = ref_atom_name
            counter = 1
            while new_name in all_mod_atom_names:
                # Изменяем только числовую часть имени
                match = re.match(r"(\D+)(\d*)", ref_atom_name)
                if match:
                    prefix = match.group(1)  # Буквенная часть
                    suffix = match.group(2)  # Числовая часть (может быть пустой)
                    new_name = f"{prefix}{int(suffix) + counter if suffix else counter}"
                else:
                    # Если имя не содержит числовой части, добавляем ее
                    new_name = f"{ref_atom_name}{counter}"
                counter += 1

            print(f'Переименовываем {mod_indx}:{ref_atom_name} -> {mod_indx}:{new_name}')
            index_rename_mod_atom = all_mod_atom_names.index(ref_atom_name)
            all_mod_atom_names[index_rename_mod_atom] = new_name
            all_mod_atom_names[mod_indx] = ref_atom_name
        else:
            # Если имени нет в модифицированной молекуле, обновляем напрямую
            all_mod_atom_names[mod_indx] = ref_atom_name


        print(f'({mod_indx}:{mod_atom_name}) -> ({ref_indx}:{ref_atom_name})')

    # Проверяем, что все имена уникальны
    name_counts = Counter(all_mod_atom_names)
    duplicates = [name for name, count in name_counts.items() if count > 1]

    if duplicates:
        print("Обнаружены повторяющиеся имена атомов в модифицированной молекуле:")
        for name in duplicates:
            print(f"Имя: {name}, количество повторений: {name_counts[name]}")
        raise ValueError("Есть повторяющиеся имена атомов. Проверьте логи.")
    else:
        print("Все имена атомов уникальны.")

    return all_mod_atom_names        


def rdkit_pdb_modification(rdkit_mol, resname='MOD', resid=1, segid='A', atom_names_list=None):
    """
    Модифицирует атрибуты атомов RDKit молекулы для сохранения в PDB формате.

    Аргументы:
        rdkit_mol: RDKit молекула, которую нужно модифицировать.
        resname: Имя остатка (3 символа, стандарт PDB).
        resid: Номер остатка.
        segid: Идентификатор сегмента (например, цепи).
        atom_names_list: Список имён атомов. Если не указан, используются стандартные.

    Возвращает:
        RDKit молекулу с обновлёнными атрибутами.
    """
    if atom_names_list is None or len(rdkit_mol.GetAtoms()) != len(atom_names_list):
        atom_names_list = [atom.GetSymbol() for atom in rdkit_mol.GetAtoms()]
        print("Длина atom_names_list не совпадает с количеством атомов в rdkit_mol. "
              "Будут использованы стандартные имена атомов.")

    atom_counter = {}

    def get_unique_atom_name(base_name, counter_dict):
        """Генерирует уникальное имя для атома на основе счётчика."""
        if base_name not in counter_dict:
            counter_dict[base_name] = 1
        else:
            counter_dict[base_name] += 1
        return f"{base_name}{counter_dict[base_name]}"

    for atom, atom_name in zip(rdkit_mol.GetAtoms(), atom_names_list):
        info = Chem.AtomPDBResidueInfo()

        # Проверка и установка уникального имени атома
        unique_atom_name = get_unique_atom_name(atom_name, atom_counter)
        info.SetName(unique_atom_name.ljust(4))  # Имя должно быть ровно 4 символа

        info.SetResidueName(resname)            # Устанавливаем имя остатка
        info.SetResidueNumber(resid)            # Устанавливаем номер остатка
        info.SetChainId(segid)                  # Устанавливаем ID сегмента

        atom.SetMonomerInfo(info)

    return rdkit_mol   

def mda_pdb_modification(mol_universe,  resname = 'MOD', resid = 1, segid = 'A'):

    mol_universe.residues.resids = resid
    mol_universe.residues.resnums = resid # warning может быть плохо 
    mol_universe.residues.resnames = resname
    mol_universe.segments.segids = segid
    mol_universe.atoms.chainIDs = segid
    return 

        
def save_aa_chem_to_pdb(rdkit_mol, path, resname = 'MOD', resid = 1, segid = 'A', make_N_root=False, atom_names_list=None):
    """
    Сохраняет молекулу в формате PDB, устанавливая корневой атом, если задан.
    Аргументы:
        rdkit_mol - RDKit.Chem молекула
        path - строка с путем к файлу и его именем для сохранения
        make_N_root - искать ли аминогруппу и делать ли ее началом молекулы
    """
    
    if make_N_root:
        root_idx = find_amino_nitrogen(rdkit_mol)
        if root_idx != -1:
            smiles_string = Chem.MolToSmiles(rdkit_mol, canonical=True, allHsExplicit=True, rootedAtAtom=root_idx)

            rdkit_mol = Chem.MolFromSmiles(smiles_string, sanitize=False)
            if rdkit_mol is None:
                raise ValueError("Ошибка при создании молекулы после перенумерации атомов")
    try:
        # Chem.SanitizeMol(rdkit_mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE, catchErrors=True)
        Chem.SanitizeMol(rdkit_mol)
    except ValueError as e:
        print("Ошибка санации:", e)
        
    # Получение пути и имени файла
    path_to_file, file_name = path_parser(path, ['pdb'])
    
    # Генерация 3D-конформации и оптимизация молекулы
    AllChem.EmbedMolecule(rdkit_mol)
    AllChem.UFFOptimizeMolecule(rdkit_mol)
    
    rdkit_mol = rdkit_pdb_modification(rdkit_mol, 
                     resname = resname, resid = resid, segid = segid, atom_names_list=atom_names_list)
    # Запись в файл
    output_path = f"{path_to_file}/{file_name}.pdb" if path_to_file else f"{file_name}.pdb"
    with open(output_path, "w") as file:
        file.write(Chem.MolToPDBBlock(rdkit_mol))
    print(f'{file_name}.pdb saved to {path_to_file or "current directory"}')

def smi_to_mol2(smi_input, output_format, file_name=None, addh=True):
    """
    Converts an SMI string or file to a molecule file format (mol2, mol, mdl).

    Args:
      smi_input: The SMI string or path to an SMI file.
      output_format: The desired output format (mol2, mol, mdl).
      file_name (optional): The desired output file name (excluding extension).

    Returns:
      None. Raises an error for invalid input or format.
    """

    valid_formats = ['mol2', 'mol', 'mdl']
    if output_format not in valid_formats:
        raise ValueError(f"Invalid output format: {output_format}")
        
    # Check if SMI is a string
    if not isinstance(smi_input, str):
        raise TypeError("Input must be a string")

    # Read SMI string from file or argument
    if smi_input.lower().endswith('.smi') or smi_input.lower().endswith('.smiles') :
        smi_str = open_smi(smi_input) 
    else:
        smi_str = smi_input.strip()
        if file_name is None:
            file_name = 'convert_mol' 

    # Print current molecule
    print(f"Current molecule: {smi_str}")

    # Create molecule, add hydrogens, and generate 3D coordinates
    mol = pybel.readstring("smi", smi_str)
    if  addh:
        mol.addh()
    mol.make3D()

    # Construct output path based on input and arguments
    output_path = file_name + '.' + output_format if file_name else smi_input.split('.')[0] + '.'+output_format

    # Write molecule to file
    mol.write(output_format, output_path, overwrite=True)
    print('output path:',output_path)
    return mol

def add_constraint(RESP_mol ,symmetric_list: list, charge_dict: dict, loc_charge: float,
                   loc_indices, constraints=psiresp.ChargeConstraintOptions()):
    """
    Создает ограничения по расчетам зарядов для одной молекулы
    Аргументы:
        RESP_mol - молеккула открытая в psiresp
        symmetric_list - лист с листами симитричных атомов
        charge_dict - словарь, где ключ - индекс атома, значение - заряд атома
        loc_indices - лист атомов сумарный заряд которых задается отдельно (loc_charge)
        loc_charge - значение сумарного заряда для подгруппы атомов (loc_indices) в молекуле 
    Возвращает:
        constraints - ограничения по расчету зарядов для молекулы
    """
    # constraints = psiresp.ChargeConstraintOptions()
    if loc_charge:
        constraints.add_charge_sum_constraint_for_molecule(RESP_mol, charge=loc_charge, indices = loc_indices)
    if symmetric_list:
        for pair_list in symmetric_list:
            constraints.add_charge_equivalence_constraint_for_molecule(RESP_mol,indices=pair_list)
    if charge_dict:
        for index, charge in charge_dict.items():
            constraints.add_charge_sum_constraint_for_molecule(RESP_mol, charge=charge, indices = index)
    return  constraints 


def prepere_constraints(mol_name, mol_chem_dict, match_dict, constraint_dict, optimize_geometry = True, charge=0,
                        conformer_generation_options = dict(n_conformer_pool=1000,
                                                            n_max_conformers=3, energy_window = 100, 
                                                            keep_original_conformer=False),
                       symmetric_atoms_are_equivalent=True, ):
    # global constraints 
    psiresp_dict = {}
    mol_chem = mol_chem_dict[mol_name]
    RESP_mol= psiresp.Molecule.from_rdkit(mol_chem,
                                          optimize_geometry = optimize_geometry, charge = charge, 
                                          conformer_generation_options= conformer_generation_options )
    psiresp_dict[mol_name] = RESP_mol
    constraints = psiresp.ChargeConstraintOptions(symmetric_atoms_are_equivalent=symmetric_atoms_are_equivalent)
    constraints = add_constraint(RESP_mol, 
                                    constraint_dict[mol_name]['symmetric_list'] , 
                                    constraint_dict[mol_name]['charge_atom_dict'], 
                                    constraint_dict[mol_name]['charge_of_monomer'],
                                    list(match_dict['mon_pol_matches'][mol_name].values()), 
                                    constraints = constraints)
    return psiresp_dict, constraints

def rename_folder(folder_name):
    """
    Генерирует новое имя папки, добавляя или увеличивая числовой суффикс.
    """
    if '/' in folder_name:
        path, folder_name = folder_name.rsplit('/',1)
        path += '/'
    else: 
        path = ''
    
    if '_' in folder_name:
        head, tail = folder_name.rsplit('_',1)
        if tail.isdigit():
            new_tail = str(int(tail)+1).zfill(2)
            new_name = f'{head}_{new_tail}'
        else: new_name = f"{head}_{tail}_01"
    else:
        head = ''
        new_name = folder_name + '_01'
    return f"{path}{new_name}"

def get_unique_folder_name(folder_name: dict or str):
    """
    Возвращает уникальное имя папки, проверяя наличие папки и изменяя имя при необходимости.
    """
    try:
        if isinstance(folder_name, dict):
            name_of_dir = str(*folder_name.keys())
        if isinstance(folder_name, str):
            name_of_dir = folder_name

        while os.path.isdir(name_of_dir):
            print(f"Директория {name_of_dir} уже существует.")
            name_of_dir = rename_folder(name_of_dir)
        print(f"Новое имя директории: {name_of_dir}")
        return name_of_dir
    except TypeError as e:
        print_red(f'Словарь должен состоять из одного элемента: \nTypeError: {e}')
    except: 
        print_red('Что-то не так')

def show_list_of_conf(directory, show_full = False):
    """
    Показывает список файлов-конформеров в директории.
    """
    files = os.listdir(directory)
    msgpack_files = [file for file in files if file.endswith(".msgpack")]
    n_conf = len(msgpack_files)
    print(f"Колличество конформеров: {n_conf}")
    if show_full:
        conf_text = '\n'.join(msgpack_files)
        print(conf_text)

        
## PSIRESP



def run_job(RESP_mol: dict or list, name_of_system, constraints, n_processes = None):
    
    
    try:
        if isinstance(RESP_mol, dict):
            RESP_list = list(RESP_mol.values())
        # global job 
        job = psiresp.Job(molecules = RESP_list, working_directory = name_of_system, n_processes = n_processes)
        job.charge_constraints = constraints
        job.run()
        return job
    except SystemExit as e:
        pass
        # print(f"SystemExit: {e}")    
    
def do_sh(name_sh: str):
    """
    Запускает .sh скрипт в указанной директории и отображает прогресс выполнения.
    """
    start_time = time.time()  # Начало отсчета времени
    done = False
    
    # Создаем поток для анимации
    t = threading.Thread(target=animate, args=(lambda: done, os.path.basename(name_sh),))
    t.start()
    
    try:
        if '/' in name_sh:
            cwd, name_sh = name_sh.rsplit('/', 1)
        else:
            cwd = None
        result = subprocess.run(["bash", name_sh], cwd=cwd, check=True)
        done = True
        t.join()
        elapsed_time = time.time() - start_time  # Подсчет прошедшего времени в минутах 
        formatted_time = format_time(elapsed_time)
        print_green(f'Время выполнения: {formatted_time}')
    except FileNotFoundError:
        done = True
        t.join()
        print(f"Не верно указана cwd: {cwd}")
    except SyntaxError:
        done = True
        t.join()
        print(f"Нет такого sh файла: {name_sh}")
    except subprocess.CalledProcessError as e:
        done = True
        t.join()
        print(f"Ошибка выполнения скрипта: {e}")

def resp_calculation(psiresp_dict, constraints, n_processes = None):
    
    name_of_system = get_unique_folder_name(psiresp_dict)

    run_job(psiresp_dict, name_of_system, constraints, n_processes = n_processes)
    time.sleep(0.7)
    show_list_of_conf(f'{name_of_system}/optimization/', False)

    do_sh(name_sh = f'{name_of_system}/optimization/run_optimization.sh')

    run_job(psiresp_dict, name_of_system, constraints, n_processes = n_processes)
    time.sleep(0.7)
    do_sh(name_sh = f'{name_of_system}/single_point/run_single_point.sh')

    job = run_job(psiresp_dict, name_of_system, constraints, n_processes = n_processes)
    # molecule = job.molecules[0]
    return job        

## Работа с листом зарядов

def make_substructure_charge_list(pol_name, charge_array, match_dict, ):
    monomer_charge = [0] * (len(match_dict['substructure'][pol_name].GetAtoms()))
    # charge_array = charge_array.round(5)
    for monomer_index, polymer_index in enumerate(match_dict['mon_pol_matches'][pol_name].values()):
        monomer_charge[monomer_index] = charge_array[polymer_index]
    monomer_charge = np.array(monomer_charge).round(4)
    return monomer_charge


def calculate_true_sum(array):
    '''
    checking the actual sum of an array
    '''
    return sum([D(f'{val}') for val in array])
        
# Финтифлюшки 

def print_green(text):
    """
    Выводит текст зеленым цветом в терминале.
    """
    print("\033[38;5;28m" + text + "\033[0m")        

def print_red(text):
    """
    Выводит текст зеленым цветом в терминале.
    """
    print("\33[31m" + text + "\33[0m") 
    
def animate(done_flag, sh_file_name):
    """
    Отображает анимацию выполнения процесса в терминале с именем sh файла.
    """
    for c in itertools.cycle(['.  ', '.. ', '...']):
        if done_flag():
            break
        sys.stdout.write(f'\rВыполнение {sh_file_name}{c}')
        sys.stdout.flush()
        time.sleep(0.5)
    # sys.stdout.write(f'\r{sh_file_name} выполнен успешно!  ')
    print_green(f'\n{sh_file_name} выполнен успешно!')

def format_time(seconds):
    """
    Форматирует время в секундах в строку формата HH:MM:SS.
    """
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02}:{mins:02}:{secs:02}"