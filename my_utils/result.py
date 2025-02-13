from hpe_library.lib_import import *

def print_and_save_result(result_dict, checkpoint_list, output_file, save_excel=True):
    import prettytable

    # PrettyTable 객체 생성 및 필드 설정
    pt = prettytable.PrettyTable()
    pt.field_names = ['Dataset', 'E1', 'E2', 'Checkpoint', 'Subset']


    # 데이터를 리스트에 수집
    data = []
    for key in checkpoint_list:
        if key not in result_dict.keys():
            print(f"Checkpoint {key}에 대한 결과가 없습니다.")
            continue
        for subset in result_dict[key].keys():
            #if 'WITH_RZ' in subset: continue
            if 'STEP' in subset: continue
            if 'TRAIN_UNIV' in subset: continue
            splited = subset.split('-')
            if 'CAM_NO_FACTOR' in splited[-1] or 'CANONICAL' in splited[-1]:
                dataset = splited[0]
            else:
                dataset = splited[0] + '-' + splited[-1]
            #print(dataset, type(dataset))
            e1 = float(result_dict[key][subset]['e1'])
            e2 = float(result_dict[key][subset]['e2'])
            data.append([dataset, e1, e2, key, subset])

    # 데이터를 Subset 1순위, checkpoint 2순위로 정렬
    data_sorted = sorted(data, key=lambda x: (x[0], len(x[3])))

    prev_dataset = ''
    for i, row in enumerate(data_sorted):
        if prev_dataset != '' and prev_dataset != row[0]:
            data_sorted[i-1].append(True)
        else:
            data_sorted[i-1].append(False)
        prev_dataset = row[0]

    # 정렬된 데이터를 테이블에 추가
    for row in data_sorted:
        pt.add_row([row[0], f'{row[1]:.2f}', f'{row[2]:.2f}', row[3], row[4]], divider=row[-1])

    # 테이블 출력
    print(pt)

    # 엑셀로 저장
    if save_excel:
        df = pd.DataFrame(pt.rows, columns=pt.field_names)
        df.to_excel(output_file, index=False)
        print(f"PrettyTable 데이터를 {output_file}로 저장했습니다!")

def seed_summary(result_dict, checkpoint_list, trainset, save_path_to_excel='', attribute_list=['seed', 'mean', 'best', 'ERR'], best_type='e1', print_table=True):
    #assert baseline in checkpoint_list, f"baseline {baseline} is not in checkpoint_list"
    field_names_e1 = []
    field_names_e2 = []

    # 데이터를 리스트에 수집
    data = {}
    total_seed_list = []
    seed_list = []
    no_seed_list = []
    for key in checkpoint_list:
        if key not in result_dict.keys():
            print(f"Checkpoint {key}에 대한 결과가 없습니다.")
            no_seed = True
        else:
            no_seed = False
        if 'seed1' in key: seed = 1
        elif 'seed2' in key: seed = 2
        elif 'seed3' in key: seed = 3
        elif 'seed4' in key: seed = 4
        else: seed = 0
        total_seed_list.append(seed)
        if 'seed' in attribute_list:
            field_names_e1.append(f'seed{seed}_e1')
            field_names_e2.append(f'seed{seed}_e2')
        if not no_seed:
            seed_list.append(seed)
            for subset in result_dict[key].keys():
                #if 'CANONICAL_PCL' in subset: continue
                if 'ROT' in subset: continue
                if 'TRAIN_UNIV' in subset: continue
                if subset not in data.keys():
                    data[subset] = {}
                data[subset][seed] = {}
                data[subset][seed]['e1'] = float(result_dict[key][subset]['e1'])
                data[subset][seed]['e2'] = float(result_dict[key][subset]['e2'])
        else:
            no_seed_list.append(seed)

    for subset in data.keys():
        for seed in no_seed_list:
            if seed not in data[subset].keys():
                data[subset][seed] = {'e1': np.NaN, 'e2': np.NaN}
        results_e1 = [data[subset][i]['e1'] for i in seed_list if not np.isnan(data[subset][i]['e1'])]
        results_e2 = [data[subset][i]['e2'] for i in seed_list if not np.isnan(data[subset][i]['e2'])]
        data[subset]['mean_e1'] = np.mean(results_e1)
        data[subset]['mean_e2'] = np.mean(results_e2)
        if best_type == 'e1':
            best_seed_for_e1 = np.argmin(results_e1)
            best_e1 = results_e1[best_seed_for_e1]
            best_e2 = results_e2[best_seed_for_e1]
        elif best_type == 'e2':
            best_seed_for_e2 = np.argmin(results_e2)
            best_e1 = results_e1[best_seed_for_e2]
            best_e2 = results_e2[best_seed_for_e2]
        elif best_type == 'all':
            best_e1 = np.min(results_e1)
            best_e2 = np.min(results_e2)
        else:
            raise ValueError(f"best_type {best_type} is not valid. Choose from ['e1', 'e2', 'all']")
        data[subset]['best_e1'] = best_e1
        data[subset]['best_e2'] = best_e2

    if 'mean' in attribute_list:
        field_names_e1.append('Average_e1')
        field_names_e2.append('Average_e2')
    if 'best' in attribute_list:
        field_names_e1.append('Best_e1')
        field_names_e2.append('Best_e2')
    #print(seed_list)

    # PrettyTable 객체 생성 및 필드 설정
    pt = prettytable.PrettyTable()
    pt.field_names = ['Trainset', 'Subset'] + field_names_e1 + field_names_e2 # ['Subset', 'seed0_e1', 'seed1_e1', 'seed2_e1', 'seed3_e1', 'seed4_e1', 'Average_e1', 'seed0_e2', 'seed1_e2', 'seed2_e2', 'seed3_e2', 'seed4_e2', 'Average_e2']
    #print(pt.field_names)
    #print(data.keys())

    rows = []
    for subset in data.keys():
        # print(subset)
        row = [trainset]
        if   subset == '3DHP-GT-CAM_NO_FACTOR-TEST_ALL_TRAIN'   : row += ['3DHP TRAIN']
        elif subset == '3DHP-GT-CAM_NO_FACTOR-TEST_TS1_6'       : row += ['3DHP TEST']
        elif subset == '3DHP-GT-CAM_NO_FACTOR-TEST_TS1_6_UNIV'  : row += ['3DHP TEST (univ)']
        elif subset == 'FIT3D-GT-CAM_NO_FACTOR-ALL_TEST'        : row += ['FIT3D ALL']
        elif subset == 'FIT3D-GT-CAM_NO_FACTOR-TR_S03'          : row += ['FIT3D S03']
        elif subset == 'FIT3D-GT-CAM_NO_FACTOR-TS_S4710'        : row += ['FIT3D S4710']
        elif subset == 'H36M-GT-CAM_NO_FACTOR'                  : row += ['H36M S9,11']
        elif subset == 'H36M-GT-CAM_NO_FACTOR-TR_S1_TS_S5678'   : row += ['H36M S5678']
        elif subset == '3DHP-GT-CAM_NO_FACTOR-TR_S1'            : row += ['3DHP TRAIN EXCEPT S1']
        elif subset == 'H36M-GT-CAM_NO_FACTOR-TR_S15_TS_S678'   : row += ['H36M S678']
        elif subset == 'H36M-GT-CAM_NO_FACTOR-TEST_ALL'         : row += ['H36M ALL']
        ##
        elif subset == '3DHP-GT-CAM_NO_FACTOR-INPUT_FROM_CANONICAL_PCL_WITH_RZ-TEST_ALL_TRAIN'  : row += ['3DHP TRAIN']
        elif subset == '3DHP-GT-CAM_NO_FACTOR-INPUT_FROM_CANONICAL_PCL_WITH_RZ-TEST_TS1_6'      : row += ['3DHP TEST']
        elif subset == '3DHP-GT-CAM_NO_FACTOR-INPUT_FROM_CANONICAL_PCL_WITH_RZ-TEST_TS1_6_UNIV' : row += ['3DHP TEST (univ)']
        elif subset == 'FIT3D-GT-CAM_NO_FACTOR-INPUT_FROM_CANONICAL_PCL_WITH_RZ-ALL_TEST'       : row += ['FIT3D ALL']
        elif subset == 'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_CANONICAL_PCL_WITH_RZ'                 : row += ['H36M S9,11']
        elif subset == 'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_CANONICAL_PCL_WITH_RZ-TR_S1_TS_S5678'  : row += ['H36M S5678']
        ##
        elif subset == '3DHP-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE-TEST_ALL_TRAIN'  : row += ['3DHP TRAIN']
        elif subset == '3DHP-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE-TEST_TS1_6'      : row += ['3DHP TEST']
        elif subset == '3DHP-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE-TEST_TS1_6_UNIV' : row += ['3DHP TEST (univ)']
        elif subset == 'FIT3D-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE-ALL_TEST'       : row += ['FIT3D ALL']
        elif subset == 'FIT3D-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE-TR_S03'         : row += ['FIT3D S03']
        elif subset == 'FIT3D-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE-TS_S4710'       : row += ['FIT3D S4710']
        elif subset == 'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE'                 : row += ['H36M S9,11']
        elif subset == 'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE-TR_S1_TS_S5678'  : row += ['H36M S5678']
        elif subset == '3DHP-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE-TR_S1'           : row += ['3DHP TRAIN EXCEPT S1']
        elif subset == 'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE-TR_S15_TS_S678'  : row += ['H36M S678']
        elif subset == 'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE-TEST_ALL'        : row += ['H36M ALL']
        ##
        elif subset == '3DHP-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE_WITH_KVIRT-TEST_ALL_TRAIN'  : row += ['3DHP TRAIN']
        elif subset == '3DHP-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE_WITH_KVIRT-TEST_TS1_6'      : row += ['3DHP TEST']
        elif subset == '3DHP-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE_WITH_KVIRT-TEST_TS1_6_UNIV' : row += ['3DHP TEST (univ)']
        elif subset == 'FIT3D-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE_WITH_KVIRT-ALL_TEST'       : row += ['FIT3D ALL']
        elif subset == 'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE_WITH_KVIRT'                 : row += ['H36M S9,11']
        elif subset == 'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE_WITH_KVIRT-TR_S1_TS_S5678'  : row += ['H36M S5678']
        ##
        elif subset == '3DHP-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE_NO_RZ-TEST_ALL_TRAIN'  : row += ['3DHP TRAIN']
        elif subset == '3DHP-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE_NO_RZ-TEST_TS1_6'      : row += ['3DHP TEST']
        elif subset == '3DHP-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE_NO_RZ-TEST_TS1_6_UNIV' : row += ['3DHP TEST (univ)']
        elif subset == 'FIT3D-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE_NO_RZ-ALL_TEST'       : row += ['FIT3D ALL']
        elif subset == 'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE_NO_RZ'                 : row += ['H36M S9,11']
        elif subset == 'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_REVOLUTE_NO_RZ-TR_S1_TS_S5678'  : row += ['H36M S5678']
        ##
        elif subset == '3DHP-GT-CAM_SCALE_FACTOR_NORM-INPUT_FROM_3D_CANONICAL_REVOLUTE-TEST_ALL_TRAIN'  : row += ['3DHP TRAIN']
        elif subset == '3DHP-GT-CAM_SCALE_FACTOR_NORM-INPUT_FROM_3D_CANONICAL_REVOLUTE-TEST_TS1_6'      : row += ['3DHP TEST']
        elif subset == '3DHP-GT-CAM_SCALE_FACTOR_NORM-INPUT_FROM_3D_CANONICAL_REVOLUTE-TEST_TS1_6_UNIV' : row += ['3DHP TEST (univ)']
        elif subset == 'FIT3D-GT-CAM_SCALE_FACTOR_NORM-INPUT_FROM_3D_CANONICAL_REVOLUTE-ALL_TEST'       : row += ['FIT3D ALL']
        elif subset == 'H36M-GT-CAM_SCALE_FACTOR_NORM-INPUT_FROM_3D_CANONICAL_REVOLUTE'                 : row += ['H36M S9,11']
        ##
        elif subset == '3DHP-GT-CAM_SCALE_FACTOR_NORM-TEST_ALL_TRAIN'  : row += ['3DHP TRAIN']
        elif subset == '3DHP-GT-CAM_SCALE_FACTOR_NORM-TEST_TS1_6'      : row += ['3DHP TEST']
        elif subset == '3DHP-GT-CAM_SCALE_FACTOR_NORM-TEST_TS1_6_UNIV' : row += ['3DHP TEST (univ)']
        elif subset == 'FIT3D-GT-CAM_SCALE_FACTOR_NORM-ALL_TEST'       : row += ['FIT3D ALL']
        elif subset == 'H36M-GT-CAM_SCALE_FACTOR_NORM'                 : row += ['H36M S9,11']
        else: row += [subset]

        if 'seed' in attribute_list:
            for seed in total_seed_list:
                row.append(data[subset][seed]['e1'])
        if 'mean' in attribute_list:
            row.append(data[subset]['mean_e1'])
        if 'best' in attribute_list:
            row.append(data[subset]['best_e1'])
        if 'seed' in attribute_list:
            for seed in total_seed_list:
                row.append(data[subset][seed]['e2'])
        if 'mean' in attribute_list:
            row.append(data[subset]['mean_e2'])
        if 'best' in attribute_list:
            row.append(data[subset]['best_e2'])
        rows.append(row)

    rows = sorted(rows, key=lambda x: (x[1]))

    # 정렬된 데이터를 테이블에 추가
    for row in rows:
        temp = [row[0], row[1]]
        temp += [f'{row[i]:.2f}' for i in range(2, len(row))]
        pt.add_row(temp)
    if print_table: print(pt)
    if save_path_to_excel != '':
        # save to excel
        df = pd.DataFrame(pt.rows, columns=pt.field_names)
        df.to_excel(save_path_to_excel, index=False)
        print(os.path.abspath(save_path_to_excel))
    return pt

def load_excel_and_print_pt(file_path, checkpoint):
    df = pd.read_excel(file_path)
    pt = prettytable.PrettyTable()
    pt.field_names = df.columns
    for row in df.values:
        pt.add_row(row)
    pt.add_column('checkpoint', [checkpoint for _ in range(len(df))])
    print(pt)
    return pt

def load_excels_merge_print_pt(checkpoints, file_paths, trainset, baseline='', attribute_list=['seed', 'mean', 'best', 'ERR'], 
                               config_folder='',
                               save_path_to_excel=''):
    if baseline != '':
        assert baseline in checkpoints, f"baseline {baseline} is not in checkpoint_list"
    dfs = [pd.read_excel(file_path) for file_path in file_paths]
    pt = prettytable.PrettyTable()
    field_names = list(dfs[0].columns) + ['checkpoint']
    #pt.field_names = field_names
    rows = []
    data = {}
    fields_seed_e1 = []
    fields_seed_e2 = []
    for df, checkpoint in zip(dfs, checkpoints):
        data[checkpoint] = {}
        for row in df.values:
            subset = row[1]
            if subset not in data[checkpoint].keys():
                data[checkpoint][subset] = {}
            for field in field_names:
                if field == 'checkpoint' or field == 'Subset': continue
                if 'seed' in field:
                    if 'e1' in field: fields_seed_e1.append(field)
                    if 'e2' in field: fields_seed_e2.append(field)
                if field not in data[checkpoint][subset].keys():
                    data[checkpoint][subset][field] = {}
                data[checkpoint][subset][field] = row[field_names.index(field)]
    fields_seed_e1 = sorted(list(set(fields_seed_e1)))
    fields_seed_e2 = sorted(list(set(fields_seed_e2)))
    print(fields_seed_e1, fields_seed_e2)

    # calculate ERR
    if baseline != '':
        for checkpoint in checkpoints:
            if checkpoint == baseline:
                for subset in data[checkpoint].keys():
                    data[checkpoint][subset]['ERR_mean_e1'] = '-'
                    data[checkpoint][subset]['ERR_best_e1'] = '-'
                    data[checkpoint][subset]['ERR_mean_e2'] = '-'
                    data[checkpoint][subset]['ERR_best_e2'] = '-'
            else:
                for subset in data[checkpoint].keys():
                    # mean e1
                    mean_e1_baseline = data[baseline][subset]['Average_e1']
                    mean_e1 = data[checkpoint][subset]['Average_e1']
                    data[checkpoint][subset]['ERR_mean_e1'] = f'{((mean_e1 - mean_e1_baseline) / mean_e1_baseline * 100):.2f}%'
                    # best e1
                    best_e1_baseline = data[baseline][subset]['Best_e1']
                    best_e1 = data[checkpoint][subset]['Best_e1']
                    data[checkpoint][subset]['ERR_best_e1'] = f'{((best_e1 - best_e1_baseline) / best_e1_baseline * 100):.2f}%'
                    # mean e2
                    mean_e2_baseline = data[baseline][subset]['Average_e2']
                    mean_e2 = data[checkpoint][subset]['Average_e2']
                    data[checkpoint][subset]['ERR_mean_e2'] = f'{((mean_e2 - mean_e2_baseline) / mean_e2_baseline * 100):.2f}%'
                    # best e2
                    best_e2_baseline = data[baseline][subset]['Best_e2']
                    best_e2 = data[checkpoint][subset]['Best_e2']
                    data[checkpoint][subset]['ERR_best_e2'] = f'{((best_e2 - best_e2_baseline) / best_e2_baseline * 100):.2f}%'

    field_names = ['Train set', 'Subset']
    if 'option' in attribute_list:
        field_names += ['CA', 'SC', 'IRC']
    if 'seed' in attribute_list: field_names += fields_seed_e1
    if 'mean' in attribute_list:
        field_names += ['Average_e1']
        if 'ERR' in attribute_list and baseline != '': field_names += ['ERR_mean_e1']
    if 'best' in attribute_list:
        field_names += ['Best_e1']
        if 'ERR' in attribute_list and baseline != '': field_names += ['ERR_best_e1']
    if 'seed' in attribute_list: field_names += fields_seed_e2
    if 'mean' in attribute_list:
        field_names += ['Average_e2']
        if 'ERR' in attribute_list and baseline != '': field_names += ['ERR_mean_e2']
    if 'best' in attribute_list:
        field_names += ['Best_e2']
        if 'ERR' in attribute_list and baseline != '': field_names += ['ERR_best_e2']

    field_names += ['checkpoint']
    pt.field_names = field_names

    rows = []
    for checkpoint in checkpoints:
        for subset in data[checkpoint].keys():
            row = [trainset, subset]
            if 'option' in attribute_list:
                # load config file
                config_file = os.path.join(config_folder, f'{checkpoint}.yaml')
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                canonical = 'v' if 'canonical' in config['model'] else ''
                # if 'input_centering' in config.keys() and config['input_centering']:
                #     input_centering =  'v'
                # else: input_centering = ''
                if 'scale_consistency' in config.keys() and config['scale_consistency']:
                    scale_consistency =  'v'
                else: scale_consistency = ''
                if 'input_residual_connection' in config.keys() and config['input_residual_connection']:
                    input_residual_connection =  'v'
                else: input_residual_connection = ''
                row += [canonical, scale_consistency, input_residual_connection]
            if 'seed' in attribute_list:
                for field in fields_seed_e1:
                    row.append(data[checkpoint][subset][field])
            if 'mean' in attribute_list:
                row.append(data[checkpoint][subset]['Average_e1'])
                if 'ERR' in attribute_list and baseline != '':  row.append(data[checkpoint][subset]['ERR_mean_e1'])
            if 'best' in attribute_list:
                row.append(data[checkpoint][subset]['Best_e1'])
                if 'ERR' in attribute_list and baseline != '':  row.append(data[checkpoint][subset]['ERR_best_e1'])
            if 'seed' in attribute_list:
                for field in fields_seed_e2:
                    row.append(data[checkpoint][subset][field])
            if 'mean' in attribute_list:
                row.append(data[checkpoint][subset]['Average_e2'])
                if 'ERR' in attribute_list and baseline != '':  row.append(data[checkpoint][subset]['ERR_mean_e2'])
            if 'best' in attribute_list:
                row.append(data[checkpoint][subset]['Best_e2'])
                if 'ERR' in attribute_list and baseline != '':  row.append(data[checkpoint][subset]['ERR_best_e2'])
            row.append(checkpoint)
            rows.append(row)

    # for df, checkpoint in zip(dfs, checkpoints):
    #     for row in df.values:
    #         rows.append(row.tolist()+[checkpoint])

    rows_sorted = sorted(rows, key=lambda x: (x[1], len(x[-1])))

    prev_subset = rows_sorted[0][0]
    divider = []
    for i, row in enumerate(rows_sorted):
        if prev_subset != row[1]:
            divider.append(True)
        else:
            divider.append(False)
        prev_subset = row[1]
    divider.append(True)
    for i, row in enumerate(rows_sorted):
        pt.add_row(row, divider=divider[i+1])
    pt.align['checkpoint'] = 'l'
    print(pt)
    if save_path_to_excel != '':
        df = pd.DataFrame(pt.rows, columns=pt.field_names)
        df.to_excel(save_path_to_excel, index=False)
        print(os.path.abspath(save_path_to_excel))
    return pt, data

def compare_two_checkpoints(model1, model2, load_path_to_excel, print_pt=True, save_path_to_excel=''):
    df = pd.read_excel(load_path_to_excel)
    data = {}
    for row in df.values:
        #print(row)
        checkpoint = row[-1]
        subset = row[0]
        #print(checkpoint, subset, mean_e1, best_e1, mean_e2, best_e2)
        if checkpoint not in data.keys():
            data[checkpoint] = {}
        for field in df.columns:
            if field == 'checkpoint' or field == 'Subset': continue
            if subset not in data[checkpoint].keys():
                data[checkpoint][subset] = {}
            data[checkpoint][subset][field] = row[df.columns.get_loc(field)]
        #data[checkpoint][subset] = {'mean_e1': mean_e1, 'best_e1': best_e1, 'mean_e2': mean_e2, 'best_e2': best_e2}

    pt = prettytable.PrettyTable()
    field_names = ['Subset']
    if 'Average_e1' in df.columns:
        field_names += ['m1_mean_e1']
        field_names += ['m2_mean_e1']
        field_names += ['ERR_mean_e1']
    if 'Average_e2' in df.columns:
        field_names += ['m1_mean_e2']
        field_names += ['m2_mean_e2']
        field_names += ['ERR_mean_e2']
    if 'Best_e1' in df.columns:
        field_names += ['m1_best_e1']
        field_names += ['m2_best_e1']
        field_names += ['ERR_best_e1']
    if 'Best_e2' in df.columns :
        field_names += ['m1_best_e2']
        field_names += ['m2_best_e2']
        field_names += ['ERR_best_e2']
    pt.field_names = field_names
    for subset in data[model1].keys():
        # subset
        row = [subset]
        # mean e1
        if 'Average_e1' in data[model1][subset].keys() and 'Average_e1' in data[model2][subset].keys():
            m1_mean_e1 = float(data[model1][subset]['Average_e1'])
            m2_mean_e1 = float(data[model2][subset]['Average_e1'])
            err_mean_e1 = (m2_mean_e1 - m1_mean_e1)/m1_mean_e1*100
            row.append(m1_mean_e1)
            row.append(m2_mean_e1)
            row.append(f'{err_mean_e1:.2f}%')
        # mean e2
        if 'Average_e2' in data[model1][subset].keys() and 'Average_e2' in data[model2][subset].keys():
            m1_mean_e2 = float(data[model1][subset]['Average_e2'])
            m2_mean_e2 = float(data[model2][subset]['Average_e2'])
            err_mean_e2 = (m2_mean_e2 - m1_mean_e2)/m1_mean_e2*100
            row.append(m1_mean_e2)
            row.append(m2_mean_e2)
            row.append(f'{err_mean_e2:.2f}%')
        # best e1
        if 'Best_e1' in data[model1][subset].keys() and 'Best_e1' in data[model2][subset].keys():
            m1_best_e1 = float(data[model1][subset]['Best_e1'])
            m2_best_e1 = float(data[model2][subset]['Best_e1'])
            err_best_e1 = (m2_best_e1 - m1_best_e1)/m1_best_e1*100
            row.append(m1_best_e1)
            row.append(m2_best_e1)
            row.append(f'{err_best_e1:.2f}%')
        # best e2
        if 'Best_e2' in data[model1][subset].keys() and 'Best_e2' in data[model2][subset].keys():
            m1_best_e2 = float(data[model1][subset]['Best_e2'])
            m2_best_e2 = float(data[model2][subset]['Best_e2'])
            err_best_e2 = (m2_best_e2 - m1_best_e2)/m1_best_e2*100
            row.append(m1_best_e2)
            row.append(m2_best_e2)
            row.append(f'{err_best_e2:.2f}%')
        pt.add_row(row)
    if print_pt: print(pt)
    if save_path_to_excel != '':
        df = pd.DataFrame(pt.rows, columns=pt.field_names)
        df.to_excel(save_path_to_excel, index=False)
    return pt