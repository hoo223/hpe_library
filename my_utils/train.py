from hpe_library.lib_import import *
# from .dataset import MyCustomDataset
# from .model import TorsoModel
# from .logger import get_logger, log_configs
# from .test_utils import readpkl
# from .dh import rotate_torso_by_R_for_batch_tensor, project_batch_tensor

# check duplicate training
def check_duplicate_training(args, blacklist=['gpu']):
    if not os.path.exists(args.out_dir): return False # if there is no folder, it is not duplicate training
    for trial in os.listdir(args.out_dir):
        try:    prev_args = readpkl(os.path.join(args.out_dir, trial, 'args.pickle'))
        except: 
            print("no args.pickle in {}".format(trial))
            continue
        same_flag = True
        for key in vars(args).keys():
            if key in blacklist: continue
            #print(key, vars(args)[key], prev_args[key])
            if vars(args)[key] != prev_args[key]:
                same_flag = False
                break
        if same_flag:
            print("same args in {}".format(trial))
            return True
    return False

# Split array by index pair
def split_array_by_idxs(array, idxs):
    array_items = []
    for i, idx in enumerate(idxs):
        array_items.append(array[:, idx[0]:idx[1]])
    return array_items

# Get available input, output candidates in dataset 
def get_input_output_candidate(args):
    targs = argparse.Namespace()

    # get input, ouput candidates with size
    input_candidate = {}
    output_candidate = {}

    # load one segment file in segment folder 
    with open(file=os.path.join(args.segment_folder, os.listdir(args.segment_folder)[0]), mode='rb') as f:
        temp_seg=pickle.load(f)

    # read input, output candidates
    for key in temp_seg[0].keys():
        if 'src' in key:
            input_candidate[key] = temp_seg[0][key].reshape(-1).shape[0]
        elif 'tar' in key:
            output_candidate[key] = temp_seg[0][key].reshape(-1).shape[0]

    targs.input_candidate = input_candidate
    targs.output_candidate = output_candidate
    args.input_candidate = input_candidate
    args.output_candidate = output_candidate

    return targs

# Load dataset
def load_dataset(args, auto_load_data=True):
    # auto_load_data: if True, automatically load data
    
    # Get input and output candidate
    targs = get_input_output_candidate(args)

    # Load train/test dataset
    training_data = MyCustomDataset(args.segment_folder, 
                                    data_type='train', 
                                    input_candidate=targs.input_candidate, 
                                    output_candidate=targs.output_candidate, 
                                    input_list=args.input_list, 
                                    output_list=args.output_list,
                                    auto_load_data=auto_load_data)
    test_data = MyCustomDataset(args.segment_folder, 
                                data_type='test', 
                                input_candidate=targs.input_candidate, 
                                output_candidate=targs.output_candidate, 
                                input_list=args.input_list, 
                                output_list=args.output_list,
                                auto_load_data=auto_load_data)
    
    # Create dataloader
    if auto_load_data:
        train_dataloader = DataLoader(training_data, batch_size=args.train_batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)
        targs.train_dataloader = train_dataloader
        targs.test_dataloader = test_dataloader


    # Update targs
    targs.training_data = training_data
    targs.test_data = test_data
        
    return args, targs

def get_num_trial(out_dir):
    # Find the last trial number
    max_trial = 0
    for fname in os.listdir(out_dir):
        if 'trial' in fname:
            num_trial = int(fname.split('_')[0].split('trial')[1])
            if num_trial > max_trial:
                max_trial = num_trial
    # if the last trial is completed
    if os.path.exists(os.path.join(out_dir, 'trial{}'.format(max_trial), 'best_model_completed.pth'.format(max_trial))): 
        num_trial = max_trial+1
    else: # if the last trial is not completed
        if max_trial > 0:
            trial_dir = os.path.join(out_dir, 'trial{}'.format(max_trial))
            shutil.rmtree(trial_dir)
            num_trial = max_trial
        else:
            num_trial = 1
    
    return num_trial

def load_model(args, targs, device):
    # Generate model
    model = TorsoModel(num_stages      = args.num_stages,
                       input_size      = targs.training_data.input_len, 
                       output_size     = targs.training_data.output_len, 
                       input_list      = targs.training_data.input_list,
                       output_list     = targs.training_data.output_list, 
                       input_idxs      = targs.training_data.input_idxs, 
                       output_idxs     = targs.training_data.output_idxs,
                       skip_connection = args.skip_connection)
    model = model.to(device)
    #model = torch.nn.DataParallel(model)
    return model

# Prepare training
def prepare_training(args, targs):
    torch.autograd.set_detect_anomaly(False)

    # Create directory for saving result
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=False)
    
    # Find the new trial number
    num_trial = get_num_trial(args.out_dir)
    print('num_trial: ', num_trial)
    
    # Create directory for saving result
    trial_dir = os.path.join(args.out_dir, 'trial{}'.format(num_trial))
    model_dir = trial_dir
    log_dir = trial_dir
    tb_dir = os.path.join(trial_dir, 'tb')
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir, exist_ok=False)

    # Set device
    assert torch.cuda.is_available(), "CUDA is not available."
    device = torch.device("cuda" if args.use_cuda else "cpu")

    # Generate model
    model = load_model(args, targs, device)

    # Criterion
    criterion = nn.MSELoss(reduction="mean")
    criterion.to(device)

    # Optimizer 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # lr_scheduler
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: args.gamma ** (step / args.decay_step)
    )

    # Make checkpoint directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=False)

    # Projection matrix tensor for reprojection loss
    args.segment_file = 'segments/' + args.segment_folder.split('segments_from_')[1].split('_with')[0] + '.pickle'
    with open(file=args.segment_file, mode='rb') as f:
        traj_segment_dataset=pickle.load(f)
    cam_proj = traj_segment_dataset['test']['cam1']['cam_param']['proj']
    cam_proj_tensor = torch.from_numpy(cam_proj).to(device).float()

    # Generate logger
    logger = get_logger(log_dir, num_trial)

    # Log configs
    log_configs(logger, args)

    # Update args, targs
    args.num_trial = num_trial
    targs.num_trial = num_trial
    targs.model = model
    targs.criterion = criterion 
    targs.optimizer = optimizer
    targs.lr_scheduler = lr_scheduler
    targs.device = device
    targs.trial_dir = trial_dir
    targs.model_dir = model_dir
    targs.log_dir = log_dir
    targs.tb_dir = tb_dir
    targs.cam_proj_tensor = cam_proj_tensor
    targs.traj_segment_dataset = traj_segment_dataset
    targs.logger = logger

    return args, targs

# Run training and test
def run(args, targs):
    # Prepare tensorboard logger.
    tblogger = SummaryWriter(targs.tb_dir)

    # Run training and test
    best_epoch = None
    test_loss_lowest = 1e10
    for epoch in range(args.epochs):
        print()

        lr = targs.lr_scheduler.get_last_lr()[0]
        targs.logger.info(f"epoch: {epoch+1}, lr: {lr:.10f}")

        # Training.
        targs.logger.info("Training...")
        train_logs = run_epoch(
            epoch        = epoch,
            model        = targs.model, 
            criterion    = targs.criterion,  
            dataloader   = targs.train_dataloader, 
            device       = targs.device,
            input_list   = targs.training_data.input_list,
            output_list  = targs.training_data.output_list, 
            input_idxs   = targs.training_data.input_idxs,
            output_idxs  = targs.training_data.output_idxs, 
            optimizer    = targs.optimizer, 
            lr_scheduler = targs.lr_scheduler,
            proj_tensor  = targs.cam_proj_tensor,
            reprojection = args.reprojection,
            train        = True,
            tblogger     = tblogger
        )
        targs.logger.info(f"train_logs: {train_logs}")
        train_loss = train_logs["total"]

        # Testing.
        targs.logger.info("Testing...")
        test_logs = run_epoch(
            epoch        = epoch,
            model        = targs.model, 
            criterion    = targs.criterion,  
            dataloader   = targs.test_dataloader, 
            device       = targs.device, 
            input_list   = targs.test_data.input_list,
            output_list  = targs.test_data.output_list, 
            input_idxs   = targs.test_data.input_idxs,
            output_idxs  = targs.test_data.output_idxs, 
            proj_tensor  = targs.cam_proj_tensor,
            reprojection = args.reprojection,
            train        = False,
            tblogger     = tblogger
        )
        targs.logger.info(f"test_logs: {test_logs}")
        test_loss = test_logs["total"]

        # Save model weight if lowest error is updated.
        if test_loss < test_loss_lowest:
            targs.logger.info("Best model is updated.")
            save_model_path = os.path.join(targs.model_dir, "best_model.pth".format(args.num_trial))
            torch.save(targs.model.state_dict(), save_model_path)
            test_loss_lowest = test_loss
            best_epoch = epoch
        else:
            targs.logger.info("Best model is not updated. (Current best: epoch {})".format(best_epoch))

        # Summary
        targs.logger.info(
            f"Summary: train_loss: {train_loss:.8f}, test_loss: {test_loss:.8f}, (best_loss): {test_loss_lowest:.8f}\n"
        )

        # Tensorboard
        for key in test_logs.keys(): # total, each output loss
            # logger.info(
            #     f"test_{key}: {test_logs[key]:.8f}"
            # )
            tblogger.add_scalar("lr", lr, epoch)
            tblogger.add_scalar("train_{}/average_loss_for_epoch".format(key), train_logs[key], epoch)
            tblogger.add_scalar("test_{}/average_loss_for_epoch".format(key), test_logs[key], epoch)

    # Best model
    targs.logger.info("Best model is epoch {} with loss {}".format(best_epoch, test_loss_lowest))
    args.save_model_path = os.path.join(targs.model_dir, "best_model_completed.pth".format(args.num_trial, best_epoch))
    os.rename(os.path.join(targs.model_dir, "best_model.pth".format(args.num_trial)), args.save_model_path)

    # Close tensorboard logger.
    tblogger.close()

    # Update args
    args.test_loss_lowest = test_loss_lowest
    args.best_epoch = best_epoch

    return args, targs

# Run training or test for an epoch.
def run_epoch(epoch,
              model,              # (torch.nn.Module): Model to train.
              criterion,          # (torch.nn.Module): Loss function.
              dataloader,         # (torch.device): CUDA device to use for training.
              device, 
              input_list,         # (list): List of input item.
              output_list,        # (list): List of output item.
              input_idxs, 
              output_idxs,        # (list): List of output indices for split.
              optimizer=None,     # (torch.optimizer): Optimizer for training
              lr_scheduler=None,  # (torch.lr_sceduler): Learning scheduler for training.
              proj_tensor=None,
              reprojection=False,
              train=True,         # (bool, optional): Whether to train the model. Defaults to True.
              tblogger=None,      # (tensorboardX.SummaryWriter, optional): Tensorboard logger. Defaults to None.
              log_steps=10,       # (int, optional): Batch interval to log losses. Defaults to 10
            ):
    """Train the model for an epoch.
    Returns:
        (dict): training/test results.
    """
    
    # Set mode
    model.train() if train else model.eval()
    mode='train' if train else 'test'

    # Initialize loss
    sum_loss, num_samples = 0, 0
    if reprojection:
        num_losses = len(output_list)+1
    else:
        num_losses = len(output_list)
    sum_loss_for_each_output = np.zeros(num_losses)

    # Run training
    batch_num = 0
    for batch in tqdm(dataloader):
        batch_num += 1
        step_num = epoch * len(dataloader) + batch_num
        input = batch[0].to(device).float()
        label = batch[1].to(device).float()
        src_torso = batch[2].to(device).float()
        tar_torso = batch[3].to(device).float()

        # Inference
        output = model(input)
        
        # Calculate losses
        label_items = split_array_by_idxs(label, output_idxs)
        losses = []

        for i in range(len(output)):
            assert len(output[i]) == len(label_items[i]), "[batch {}] batch_size is not same. {} != {}, input {}".format(batch_num, len(output[i]), len(label_items[i]), len(input))
            loss = criterion(output[i], label_items[i])
            losses.append(loss)

        if reprojection:
            if 'tar_delta_rotvec' in output_list and 'tar_delta_theta' in output_list:
                pred_delta_point, pred_delta_rotvec, pred_delta_theta = output
                pred_delta_point_batch = pred_delta_point.reshape(-1, 1, 3).repeat(1, 5, 1)
                pred_delta_rot = torch.tensor(Rotation.from_rotvec(pred_delta_rotvec*pred_delta_theta).as_matrix()).float().to(device)
                pred_torso = rotate_torso_by_R_for_batch_tensor(src_torso, pred_delta_rot) + pred_delta_point_batch
                pred_torso_projected = project_batch_tensor(pred_torso, proj_tensor)

            elif 'tar_delta_torso' in output_list:
                pred_delta_torso = output[0]
                pred_torso = src_torso + pred_delta_torso.reshape(-1, 5, 3)

            elif 'tar_delta_point' in output_list and 'tar_delta_quat' in output_list:
                pred_delta_point, pred_delta_quat = output
                pred_delta_point_batch = pred_delta_point.reshape(-1, 1, 3).repeat(1, 5, 1)
                pred_delta_rot = torch.tensor(Rotation.from_quat(pred_delta_quat.cpu().detach().numpy()).as_matrix()).float().to(device)
                pred_torso = rotate_torso_by_R_for_batch_tensor(src_torso, pred_delta_rot) + pred_delta_point_batch
                pred_torso_projected = project_batch_tensor(pred_torso, proj_tensor)
            elif 'tar_delta_point' in output_list:
                pred_delta_point = output[0]
                pred_delta_point_batch = pred_delta_point.reshape(-1, 1, 3).repeat(1, 5, 1)
                pred_torso = src_torso + pred_delta_point_batch
                pred_torso_projected = project_batch_tensor(pred_torso, proj_tensor)
            elif 'tar_torso' in output_list:
                pred_torso = output[0].reshape(-1, 5, 3)
                pred_torso_projected = project_batch_tensor(pred_torso, proj_tensor) 
            else:
                print("No reprojection loss")

            # target
            tar_torso_projected = project_batch_tensor(tar_torso, proj_tensor)

            reprojection_loss = criterion(pred_torso_projected[:, :, :2]/1000, tar_torso_projected[:, :, :2]/1000)
            losses.append(reprojection_loss)

        loss = sum(losses)

        # Backpropagation
        if train:
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        # Update loss
        batch_size = len(input)
        num_samples += batch_size
        for i in range(num_losses):
            sum_loss_for_each_output[i] += losses[i].item() #* batch_size
        
        # Log to tensorboard
        if (tblogger is not None) and (batch_num % log_steps == 0):
            tblogger.add_scalar("{}_total/average_loss_per_step".format(mode), loss.item()/batch_size, step_num)
            for i in range(len(output_list)): 
                tblogger.add_scalar("{}_{}/average_loss_per_step".format(mode, output_list[i]), losses[i].item()/batch_size, step_num)
            if reprojection:
                tblogger.add_scalar("{}_reprojection/average_loss_per_step".format(mode), losses[-1].item()/batch_size, step_num)

    # Calculate average loss
    sum_loss = sum_loss_for_each_output.sum()
    average_loss = sum_loss / num_samples
    metrics = {"total": average_loss}
    for i in range(len(output_list)):
        metrics[output_list[i]] = sum_loss_for_each_output[i] / num_samples
    if reprojection:
        metrics['reprojection'] = sum_loss_for_each_output[-1] / num_samples

    return metrics

# Store args in pickle file
def save_args(args, trial_dir):
    if not isinstance(args, dict): dict_args = vars(args)
    else: dict_args = args
    with open(os.path.join(trial_dir, 'args.pickle'), 'wb') as f:
        pickle.dump(dict_args, f)
        
def load_args(out_dir, num_trial):
    return readpkl(os.path.join(out_dir, 'trial{}'.format(num_trial), 'args.pickle'))

# Load best model of specific trial
def load_best_model(args, targs, eval=True, device=None, model=None):
    if model is None:
        model = load_model(args, targs, device)
    model.load_state_dict(torch.load(args.save_model_path))
    if eval:
        model.eval()
    else:
        model.train()
    return model