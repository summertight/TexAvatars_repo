import torch

def interp_face_scale(face_scale, size):

    repeat_count, remainder = size // len(face_scale), size % len(face_scale)

    base_indices = torch.arange(len(face_scale)).repeat_interleave(repeat_count)
    extra_indices = torch.arange(remainder)

    result = torch.cat((base_indices, extra_indices))
    return face_scale[result]



def interp_face_quat(face_orien_quat, size):
    repeat_count, remainder = size // len(face_orien_quat), size % len(face_orien_quat)

    base_indices = torch.arange(len(face_orien_quat)).repeat_interleave(repeat_count)
    extra_indices = torch.arange(remainder)

    result = torch.cat((base_indices, extra_indices))
    return face_orien_quat[result]

def interp_face_mat(face_orien_mat, size):
    repeat_count, remainder = size // len(face_orien_mat), size % len(face_orien_mat)

    base_indices = torch.arange(len(face_orien_mat)).repeat_interleave(repeat_count)
    extra_indices = torch.arange(remainder)

    result = torch.cat((base_indices, extra_indices))
    return face_orien_mat[result]

def interp_face_xyz(face_xyz, size):
    repeat_count, remainder = size // len(face_xyz), size % len(face_xyz)

    base_indices = torch.arange(len(face_xyz)).repeat_interleave(repeat_count)
    extra_indices = torch.arange(remainder)

    result = torch.cat((base_indices, extra_indices))
    return face_xyz[result]