import torch


def collate_tensors(batch: list) -> torch.Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def mld_collate(batch: list) -> dict:
    notnone_batches = [b for b in batch if b is not None]
    notnone_batches.sort(key=lambda x: x[3], reverse=True)

    batch_len = len(notnone_batches[0])

    if batch_len == 8:
        adapted_batch = {
            "motion":
            collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
            "text": [b[2] for b in notnone_batches],
            "length": [b[5] for b in notnone_batches],
            "word_embs":
            collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
            "pos_ohot":
            collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
            "text_len":
            collate_tensors([torch.tensor(b[3]) for b in notnone_batches]),
            "tokens": [b[6] for b in notnone_batches],
            
        }
    if batch_len == 9:
        adapted_batch = {
            "word_embs":
            collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
            "pos_ohot":
            collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
            "text": [b[2] for b in notnone_batches],
            "text_len":
            collate_tensors([torch.tensor(b[3]) for b in notnone_batches]),
            "motion":
            collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
            "length": [b[5] for b in notnone_batches],
            "tokens": [b[6] for b in notnone_batches],
            "hint_2d": collate_tensors([torch.tensor(b[7]).float() for b in notnone_batches])

        }
    if batch_len == 15:
        adapted_batch = {
            "word_embs":
            collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
            "pos_ohot":
            collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
            "text": [b[2] for b in notnone_batches],
            "text_len":
            collate_tensors([torch.tensor(b[3]) for b in notnone_batches]),
            "motion":
            collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
            "length": [b[5] for b in notnone_batches],
            "tokens": [b[6] for b in notnone_batches],
            "hint_2d_a": collate_tensors([torch.tensor(b[7]).float() for b in notnone_batches]),
            "hint_2d_b": collate_tensors([torch.tensor(b[8]).float() for b in notnone_batches]),
            "pose_2d": collate_tensors([torch.tensor(b[9]).float() for b in notnone_batches]),
            "pose_3d": collate_tensors([torch.tensor(b[10]).float() for b in notnone_batches]),
            "pose_mask": collate_tensors([torch.tensor(b[11]).float() for b in notnone_batches]),
            "rotation": collate_tensors([torch.tensor(b[12]).float() for b in notnone_batches]),
            "pose_2d_b": collate_tensors([torch.tensor(b[13]).float() for b in notnone_batches])
        }
    elif batch_len == 10:
        adapted_batch = {
            "word_embs":
            collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
            "pos_ohot":
            collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
            "text": [b[2] for b in notnone_batches],
            "text_len":
            collate_tensors([torch.tensor(b[3]) for b in notnone_batches]),
            "motion":
            collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
            "length": [b[5] for b in notnone_batches],
            "tokens": [b[6] for b in notnone_batches],
            # "V": [b[7] for b in notnone_batches],
            "pose":collate_tensors([torch.tensor(b[7]).float() for b in notnone_batches]),
            "pose_mask": collate_tensors([torch.tensor(b[8]).bool() for b in notnone_batches]),
            # "motion_idx": [b[9] for b in notnone_batches],
        }
    elif batch_len == 11:
        adapted_batch = {
            "word_embs":
            collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
            "pos_ohot":
            collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
            "text": [b[2] for b in notnone_batches],
            "text_len":
            collate_tensors([torch.tensor(b[3]) for b in notnone_batches]),
            "motion":
            collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
            "length": [b[5] for b in notnone_batches],
            "tokens": [b[6] for b in notnone_batches],
            "V": [b[7] for b in notnone_batches],
            "pose":collate_tensors([torch.tensor(b[8]).float() for b in notnone_batches]),
            "motion_idx": [b[9] for b in notnone_batches],
        }
    # elif batch_len == 12:
    #     adapted_batch = {
    #         "word_embs":
    #         collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
    #         "pos_ohot":
    #         collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
    #         "text": [b[2] for b in notnone_batches],
    #         "text_len":
    #         collate_tensors([torch.tensor(b[3]) for b in notnone_batches]),
    #         "motion":
    #         collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
    #         "length": [b[5] for b in notnone_batches],
    #         "tokens": [b[6] for b in notnone_batches],
    #         "V": [b[7] for b in notnone_batches],
    #         "pose":collate_tensors([torch.tensor(b[8]).float() for b in notnone_batches]),
    #         "motion_idx": [b[9] for b in notnone_batches],
    #         "pose_mask": collate_tensors([torch.tensor(b[10]).bool() for b in notnone_batches]),
    #     }
    elif batch_len == 18:
        adapted_batch = {
            "motion":
            collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
            "text": [b[2] for b in notnone_batches],
            "length": [b[5] for b in notnone_batches],
            "word_embs":
            collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
            "pos_ohot":
            collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
            "text_len":
            collate_tensors([torch.tensor(b[3]) for b in notnone_batches]),
            "tokens": [b[6] for b in notnone_batches],
            "V": [b[7] for b in notnone_batches],
          
            "pose":collate_tensors([torch.tensor(b[8]).float() for b in notnone_batches]),
            "motion_idx": [b[9] for b in notnone_batches],

            "text_2": [b[10] for b in notnone_batches],
            "motion_2":
            collate_tensors([torch.tensor(b[11]).float() for b in notnone_batches]),
            "length_2": [b[12] for b in notnone_batches],
            "V_2": [b[13] for b in notnone_batches],
            "pose_2":collate_tensors([torch.tensor(b[14]).float() for b in notnone_batches]),
            "motion_idx_2": [b[15] for b in notnone_batches],
            "hint2": collate_tensors([torch.tensor(b[16]).float() for b in notnone_batches])
        }

    # collate trajectory
    if notnone_batches[0][-1] is not None:
        adapted_batch['hint'] = collate_tensors([torch.tensor(b[-1]).float() for b in notnone_batches])

    return adapted_batch
