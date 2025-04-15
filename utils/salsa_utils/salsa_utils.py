# Todo: Implemet data loader preprocess and Salsa SMPL to HumanML3D representation.
from glob import glob

from smplx import SMPLX
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import HumanMl3D_functions as HM3D_F

import lmdb
import pyarrow
from scipy.interpolate import interp1d
from moviepy.editor import VideoFileClip
import librosa
import soundfile as sf
import torchaudio
#Todo: ---------------------<HumanML3D-Functions>--------------------------------
#Todo: --------------------------------------------------------------------------


#Todo: ---------------------</HumanML3D-Functions>--------------------------------
#Todo: --------------------------------------------------------------------------
def salsa_smplx_to_pos3d(data):
    smplx = None
    joints_num = 22 # to be consistent with HumanML3D
    # smplx = SMPLX(model_path='/localhome/cza152/Desktop/Duolando/smplx/models/smplx',
    #                betas=data['betas'][:, :10], gender=data['meta']['gender'], \
    #     batch_size=len(data['betas']), num_betas=10, use_pca=False, use_face_contour=True, flat_hand_mean=True)
    frames = data['poses'].shape[0]
    b = np.repeat(data['betas'][:10], frames).reshape((frames, 10))
    smplx = SMPLX(model_path='SMPLX_DEP\\models_lockedhead\\smplx', betas=b,
                  gender=np.array2string(data['gender'])[1:-1], \
                  batch_size=len(b), num_betas=10, use_pca=False, use_face_contour=True, flat_hand_mean=True)

    keypoints3d = smplx.forward(
        # global_orient=torch.from_numpy(data['global_orient']).float(),

        global_orient=torch.from_numpy(data['poses'][:, :3], ).float(),
        body_pose=torch.from_numpy(data['poses'][:, 3:66]).float(),
        jaw_pose=torch.from_numpy(data['poses'][:, 66:69]).float(),
        leye_pose=torch.from_numpy(data['poses'][:, 69:72]).float(),
        reye_pose=torch.from_numpy(data['poses'][:, 72:75]).float(),
        left_hand_pose=torch.from_numpy(data['poses'][:, 75:120]).float(),
        right_hand_pose=torch.from_numpy(data['poses'][:, 120:]).float(),
        transl=torch.from_numpy(data['trans']).float(),  # transl=torch.from_numpy(data['transl']).float(),
        # betas=torch.from_numpy(data['betas'][:10]).float()
        betas=torch.from_numpy(b).float()
    ).joints.detach().numpy()[:, :joints_num]
    # ours is (N, 144, 3)
    #
    # nframes = keypoints3d.shape[0]
    # keypoints3d = keypoints3d.reshape(nframes, -1)
    #
    # # Calculate relative offset with respect to root
    # root = keypoints3d[:, :3]  # the root
    # keypoints3d = keypoints3d - np.tile(root, (1, 25))
    # # keypoints3d[:, :3] = root

    # 1. after getting Jtr
    trans_matrix = np.array([[1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0],
                             [0.0, 1.0, 0.0]])
    keypoints3d = np.dot(keypoints3d, trans_matrix)

    keypoints3d = interp1d(np.linspace(0, 1, len(keypoints3d)), keypoints3d, axis=0)(
        np.linspace(0, 1, int(len(keypoints3d) * 20 / 30)))


    # Expected shape is (N, 52, 3) or (N, 24, 3)


    return keypoints3d



def salsa_smplx_to_rotmat(data):
    smpl_poses, smpl_trans = data['poses'], data['trans']

    nframes = smpl_poses.shape[0]
    njoints = 55

    r = R.from_rotvec(smpl_poses.reshape([nframes * njoints, 3]))
    rotmat = r.as_matrix().reshape([nframes, njoints, 3, 3])

    rotmat = np.concatenate([
        smpl_trans,
        rotmat.reshape([nframes, njoints * 3 * 3])
    ], axis=-1)

    rotmat = interp1d(np.linspace(0, 1, len(rotmat)), rotmat, axis=0)(
        np.linspace(0, 1, int(len(rotmat) * 20 / 30)))

    nframes = rotmat.shape[0]
    return rotmat.reshape(nframes, -1)


def Salsa2HM3D():
    sample_path = ''

    salsa_smplx_to_pos3d()

def audio_from_mp4(vide_path, audio_output_path):

    try:
        if not os.path.exists(audio_output_path):
            video = VideoFileClip(vide_path)
            video.audio.write_audiofile(audio_output_path)
            print(f"Extracted audio from {vide_path} -> {audio_output_path}")

        y, sr = torchaudio.load(audio_output_path)
        return y.cpu().numpy(), sr
    except Exception as e:
        print(f"Failed to process {vide_path}: {e}")


joints_num = 22
def read_all_salsa(base_path):

    out_path = os.path.join(base_path, 'lmdb_Salsa')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    map_size = 1024 * 12  # in MB
    map_size <<= 20  # in B
    db = [lmdb.open(os.path.join(out_path, 'lmdb_train'), map_size=map_size),
          lmdb.open(os.path.join(out_path, 'lmdb_test'), map_size=map_size)]

    # delete existing files
    for i in range(2):
        with db[i].begin(write=True) as txn:
            txn.drop(db[i].open_db())

    # all_poses = []
    all_keypoints3d = []
    all_rotmat = []


    smpl_root = 'S:\Payam\Dance_Salsa_SFU\delivery_241121\delivery_241121'
    pos3d_root = './salsa_data/motion/pos3d'
    rotmat_root = './salsa_data/motion/rotmat'

    mp4_root = 'S:\Payam\Dance_Salsa_SFU\salsa project\salsa project\Animations'
    mp4_files = glob(os.path.join(mp4_root, '*.mp4'))
    smplx2mp4_map = {os.path.basename(f).replace('.mp4', ''): f for f in mp4_files}

    os.makedirs(pos3d_root, exist_ok=True)
    os.makedirs(rotmat_root, exist_ok=True)
    v_i = 0
    ex_fps = 20
    fps = 30

    for folder in os.listdir(smpl_root):
        # if v_i>2:
        #     break
        print(folder)
        smplx_folder = os.path.join(smpl_root, folder)
        pos3d_folder = os.path.join(pos3d_root, folder)
        rotmat_folder = os.path.join(rotmat_root, folder)
        if not os.path.exists(pos3d_folder):
            os.mkdir(pos3d_folder)
        if not os.path.exists(rotmat_folder):
            os.mkdir(rotmat_folder)
        for takes_folder in os.listdir(smplx_folder):
            if v_i > 2 : break
            for file in os.scandir(os.path.join(smplx_folder, takes_folder)):
                if not file.name.endswith('.npz'): continue
                # if "Pair2" not in file.name: continue



                mp4_path = smplx2mp4_map['_'.join(file.name.split('_')[:5])]
                audio_path = mp4_path.replace('mp4', 'wav')
                audio_y, audio_sr = audio_from_mp4(mp4_path, audio_path)


                loaded = np.load(file.path, allow_pickle=True)
                # Todo: HumanML3D/ raw_pose_processing.ipynb --> Done!
                keypoints3d = salsa_smplx_to_pos3d(loaded)
                rotmat = salsa_smplx_to_rotmat(loaded)
                # HumanML3D Representation
                (data, ground_positions,
                 positions, l_velocity) = HM3D_F.process_file(keypoints3d,
                                                              0.002)
                rec_ric_data = HM3D_F.recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)
                HML3D_New_Joints = rec_ric_data.squeeze().numpy() # N, 22, 3
                HML3D_New_Joints_Vec = data


                # test:
                # import os
                # pjoin = os.path.join
                # np.save(pjoin(out_path, 'SFU_SALSA_EXAMPLE.npy'), HML3D_New_Joints_Vec[:1000])

                # np.save(pjoin(save_dir1, source_file), rec_ric_data.squeeze().numpy())
                # np.save(pjoin(save_dir2, source_file), data)
                # Todo: motion_representation.ipynb
                # Done!

                # # Sanity Check:
                # save_path = '1.gif'
                # # HM3D_F.plot_3d_motion(save_path, kinematic_chain, New_Joints[:20], title="None", fps=20, radius=4)
                # ARGUS = HML3D_New_Joints[:400], save_path, 'title'
                # HM3D_F.plot_3d_motion_Payam(ARGUS)

                # for v_i, bvh_file in enumerate(bvh_files):
                name = os.path.split(file.name)[1][:-4]
                print(name)

                # process
                clips = [{'vid': name, 'clips': []},  # train
                         {'vid': name, 'clips': []}]  # validation

                # split
                if v_i == 0:
                    dataset_idx = 1  # validation
                else:
                    dataset_idx = 0  # train

                # save subtitles and skeletons
                poses = np.asarray(rotmat, dtype=np.float16)
                clips[dataset_idx]['clips'].append(
                    {'raw_euler_poses': loaded['poses'],
                     'raw_trans': loaded['trans'],
                     'keypoints3d': keypoints3d,
                     'rotmat': rotmat,
                     'HML3D_joints': HML3D_New_Joints,
                     'HML3D_joints_vec': HML3D_New_Joints_Vec,
                     'audio_raw': audio_y,
                     'audio_sr': audio_sr
                     # Todo: add motioncodes here? No, we add it after windowing.
                     })

                # write to db
                for i in range(2):
                    with db[i].begin(write=True) as txn:
                        if len(clips[i]['clips']) > 0:
                            k = '{:010}'.format(v_i).encode('ascii')
                            v = pyarrow.serialize(clips[i]).to_buffer()
                            txn.put(k, v)

                # # all_poses.append(poses)
                # all_keypoints3d.append(keypoints3d)
                # all_rotmat.append(rotmat)
                v_i += 1

        # close db
    for i in range(2):
        db[i].sync()
        db[i].close()

read_all_salsa('Salsa_Temp')