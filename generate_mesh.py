import blenderproc as bpro
import os
import random
import numpy as np
import glob
from tqdm import tqdm

from blenderproc.python.utility.SetupUtility import SetupUtility
from blenderproc.python.utility.Utility import resolve_path
from blenderproc.python.loader.AMASSLoader import _AMASSLoader


def get_pose_parameters(supported_mocap_datasets, num_betas, used_sub_dataset_id,
                            used_subject_id, used_sequence_id,
                            used_frame_id):
        """ Extract pose and shape parameters corresponding to the requested pose from the database to be
        processed by the parametric model

        :param supported_mocap_datasets: A dict which maps sub dataset names to their paths.
        :param num_betas: Number of body parameters
        :param used_sub_dataset_id: Identifier for the sub dataset, the dataset which the human pose object
                                    should be extracted from.
        :param used_subject_id: Type of motion from which the pose should be extracted, this is dataset
                                dependent parameter.
        :param used_sequence_id: Sequence id in the dataset, sequences are the motion recorded to represent
                                 certain action.
        :param used_frame_id: Frame id in a selected motion sequence. If none is selected a random one is picked
        :return: tuple of arrays contains the parameters. Type: tuple
        """
        # This import is done inside to avoid having the requirement that BlenderProc depends on torch
        #pylint: disable=import-outside-toplevel
        import torch
        #pylint: enable=import-outside-toplevel
        # print(used_sub_dataset_id)
        # check if the sub_dataset is supported
        if used_sub_dataset_id in supported_mocap_datasets:
            # get path from dictionary
            sub_dataset_path = supported_mocap_datasets[used_sub_dataset_id]
            # concatenate path to specific
            if not used_subject_id:
                # if none was selected
                possible_subject_ids = glob.glob(os.path.join(sub_dataset_path, "*"))
                possible_subject_ids.sort()
                if len(possible_subject_ids) > 0:
                    used_subject_id_str = os.path.basename(random.choice(possible_subject_ids))
                else:
                    raise FileNotFoundError(f"No subjects found in folder: {sub_dataset_path}")
            else:
                used_subject_id_str = f"{int(used_subject_id):02d}"

            if used_sequence_id < 0:
                # if no sequence id was selected
                # assert False
                possible_sequence_ids = glob.glob(os.path.join(sub_dataset_path, used_subject_id_str, "*"))
                possible_sequence_ids.sort()
                if len(possible_sequence_ids) > 0:
                    used_sequence_id = os.path.basename(random.choice(possible_sequence_ids))
                    used_sequence_id = used_sequence_id[used_sequence_id.find("_")+1:used_sequence_id.rfind("_")]
                else:
                    raise FileNotFoundError(f"No sequences found in folder: "
                                            f"{os.path.join(sub_dataset_path, used_subject_id_str)}")
            subject_path = os.path.join(sub_dataset_path, used_subject_id_str)
            used_subject_id_str_reduced = used_subject_id_str[:used_subject_id_str.find("_")] \
                if "_" in used_subject_id_str else used_subject_id_str
            sequence_path = os.path.join(subject_path, used_subject_id_str_reduced +
                                         f"_{int(used_sequence_id):02d}_poses.npz")
            if os.path.exists(sequence_path):
                # load AMASS dataset sequence file which contains the coefficients for the whole motion sequence
                sequence_body_data = np.load(sequence_path)
                # get the number of supported frames
                no_of_frames_per_sequence = sequence_body_data['poses'].shape[0]
                if used_frame_id < 0:
                    frame_id = random.randint(0, no_of_frames_per_sequence - 1)  # pick a random id
                else:
                    frame_id = used_frame_id
                # Extract Body Model coefficients
                if frame_id in range(0, no_of_frames_per_sequence):
                    # use GPU to accelerate mesh calculations
                    comp_device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
                    # parameters that control the body pose
                    # refer to http://files.is.tue.mpg.de/black/papers/amass.pdf, Section 3.1 for more
                    # information about the parameter representation and the below chosen values
                    pose_body = torch.Tensor(sequence_body_data['poses'][frame_id:frame_id + 1, 3:66]).to(comp_device)
                    # parameters that control the body shape
                    betas = torch.Tensor(sequence_body_data['betas'][:num_betas][np.newaxis]).to(comp_device)
                    return pose_body, betas, used_sequence_id, frame_id
                raise RuntimeError(f"Requested frame id is beyond sequence range, for the selected sequence, choose "
                                   f"frame id within the following range: [0, {no_of_frames_per_sequence}]")
            raise RuntimeError(f"Invalid sequence/subject: {used_subject_id} category identifiers, please choose a "
                               f"valid one. Used path: {sequence_path}")
        raise RuntimeError(f"The requested mocap dataset is not yest supported, please choose another one "
                           f"from the following supported datasets: {list(supported_mocap_datasets.keys())}")



def write_body_mesh_to_obj_file(body_representation, faces, temp_dir, 
                                sub_dataset_id, body_model_gender, subject_id, sequence_id, frame_id):
        """ Write the generated pose as obj file on the desk.

        :param body_representation: parameters generated from the BodyModel model which represent the obj
                                     pose and shape. Type: torch.Tensor
        :param faces: face parametric model which is used to generate the face mesh. Type: numpy.array
        :param temp_dir: Path to the folder in which the generated pose as obj will be stored
        :return: path to generated obj file. Type: string.
        """
        obj_file_name = f'{sub_dataset_id}_{body_model_gender}_{subject_id}_{sequence_id}_{frame_id}.obj'
        # Write to an .obj file
        outmesh_path = os.path.join(temp_dir, obj_file_name)
        with open(outmesh_path, 'w', encoding="utf-8") as fp:
            fp.write("".join([f'v {v[0]:f} {v[1]:f} {v[2]:f}\n' for v in
                              body_representation.v[0].detach().cpu().numpy()]))
            fp.write("".join([f'f {f[0]} {f[1]} {f[2]}\n' for f in faces + 1]))
        return outmesh_path


def load_AMASS(data_path, sub_dataset_id, temp_dir=None, body_model_gender=None, 
               subject_id="", sequence_id=-1, frame_id=-1, num_betas=10, num_dmpls=8):

    if body_model_gender is None:
        body_model_gender = random.choice(["male", "female", "neutral"])

    # Install required additonal packages
    SetupUtility.setup_pip(["git+https://github.com/abahnasy/smplx",
                            "git+https://github.com/abahnasy/human_body_prior"])

    # Get the currently supported mocap datasets by this loader
    taxonomy_file_path = resolve_path(os.path.join(data_path, "taxonomy.json"))
    supported_mocap_datasets = _AMASSLoader.get_supported_mocap_datasets(taxonomy_file_path, data_path)

    # selected_obj = self._files_with_fitting_ids
    pose_body, betas, sequence_id, frame_id = get_pose_parameters(supported_mocap_datasets, num_betas, sub_dataset_id, subject_id, sequence_id, frame_id)
    # print(sequence_id)
    # assert False
    # load parametric Model
    body_model, faces = _AMASSLoader.load_parametric_body_model(data_path, body_model_gender, num_betas, num_dmpls)
    # Generate Body representations using SMPL model
    # print(pose_body.shape, betas.shape)
    # assert False

    # pose_body = torch.zeros(1, 63).float().cuda()
    # betas = torch.zeros(1, 10).float().cuda()

    body_repr = body_model(pose_body=pose_body, betas=betas)

    # print(body_repr, faces)
    # assert False
    # Generate .obj file represents the selected pose
    generated_obj = write_body_mesh_to_obj_file(
         body_repr, faces, temp_dir, 
         sub_dataset_id, body_model_gender, subject_id, sequence_id, frame_id
    )
    # print(generated_obj)
    # assert False


if __name__ == '__main__':

    amass_root = './resources/AMASS'
    data_root = './resources/AMASS/CMU/CMU'
    save_dir = './generated_meshes'

    sequence = {}

    cmus = ['07', '08', '09', '17', '35', '36', '37', '38', '45', '46', '47', '78', '91']

    # sequence = {
    #     '07': [1, 12], '08': [1, 10], '09': [1, 12], '17': [1, 10], '35': [1, 34], 
    #     '36': [1, 36], '37': [1, 1], '38': [1, 4], '45': [1, 1], '46': [1, 1], 
    #     '47': [1, 1], '78': [2, 35], '91': [1, 62]
    #     }

    for cmu in cmus:
        sequence[cmu] = [int(x.split('_')[1]) for x in os.listdir(os.path.join(data_root, cmu))]

    frames = {
        '07': 120, '08': 120, '09': 120, '17': 120, '35': 120, '36': 120, '37': 120, 
        '38': 120, '45': 120, '46': 120, '47': 120, '78': 97, '91': 120
        }

    genders = ['male', 'female']

    os.makedirs(save_dir, exist_ok=True)

    count = 0

    for cmu in tqdm(cmus):
        # print(cmu)
        for gender in genders:
            for seq in sequence[cmu]:
                # for frame in range(frames[cmu])[::50]:
                # print(cmu, seq, frame)
                count += 1
                load_AMASS(
                    amass_root,
                    sub_dataset_id="CMU",
                    body_model_gender=gender,
                    subject_id=cmu,
                    sequence_id=seq,
                    # frame_id=frame,
                    temp_dir=save_dir
                )
                # assert False

    print(count)