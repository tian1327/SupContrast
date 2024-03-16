"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        
        # print('features.shape=', features.shape) # 256, 2, 512
        

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # print('mask.shape=', mask.shape) # 256, 256

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print('contrast_feature.shape=', contrast_feature.shape) # 512, 512
        # print('contrast_count=', contrast_count) # 2
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # print('anchor_feature.shape=', anchor_feature.shape) # 512, 512
        # print('anchor_count=', anchor_count) # 2

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # print('anchor_dot_contrast.shape=', anchor_dot_contrast.shape) # 512, 512


        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # print('logits.shape=', logits.shape) # 512, 512
        # print('logits_max', logits_max) # 10, 10, ..., 10

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # print('mask.shape=', mask.shape) # 256, 256

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # print('mask.shape=', mask.shape)
        # print('logits_mask.shape=', logits_mask.shape)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class FASupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(FASupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels, sources, mask=None):
        """Compute loss for model. 

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            source: 1 indicates fewshot annotation, 0 indicates retrieved images
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        # print('features.shape=', features.shape) # 256, 2, 512
        
        batch_size = features.shape[0]
        has_fewshot = torch.sum(sources) > 0
        fewshot_ct = torch.sum(sources)
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # print('labels.shape=', labels.shape) # 256, 1
            # print(labels)
            # set labels to [0, 1, 1, 1]
            # labels = torch.tensor([0, 1, 1, 1]).view(4, 1)
            mask = torch.eq(labels, labels.T).float().to(device) # symmetric mask

            # here we use the resource to set the retrived images to have mask value of 0
            # resource = 0 indicates the retrieved images, 1 indicates the fewshot annotation
            # the resource is a vector of shape [bsz, 1]
            # print('mask.shape=', mask.shape) # 256, 256
            # print(mask)
            # print('sources.shape=', sources.shape) # 256, 1
            # print(sources)
            # sources = torch.tensor([0, 0, 1, 0]).view(4, 1)
            if has_fewshot:
                # for each row whose value in the sources, if the source is 0, set the entire row to 0
                mask[sources.squeeze() == 0, :] = 0
                        
        else:
            mask = mask.float().to(device)
        # print('mask.shape=', mask.shape) # 256, 256        

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print('contrast_feature.shape=', contrast_feature.shape) # 512, 512
        # print('contrast_count=', contrast_count) # 2
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # print('anchor_feature.shape=', anchor_feature.shape) # 512, 512
        # print('anchor_count=', anchor_count) # 2

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # print('anchor_dot_contrast.shape=', anchor_dot_contrast.shape) # 512, 512

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # print('logits.shape=', logits.shape) # 512, 512
        # print('logits_max', logits_max) # 10, 10, ..., 10

        # tile mask
        # print('mask.shape=', mask.shape) # 256, 256
        # print('anchor_count=', anchor_count) # 2
        # print('contrast_count=', contrast_count) # 2
        mask = mask.repeat(anchor_count, contrast_count)
        # print('mask.shape=', mask.shape) # 512, 512

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), # create a tensor with the same shape as mask, filled with 1
            1, # dimension along which to scatter
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), # index vector
            0 # replace the value at the index with 0, this sets diagonal values to 0
        )
        mask = mask * logits_mask
        # print('mask.shape=', mask.shape) # 512, 512
        # print('logits_mask.shape=', logits_mask.shape) # 512, 512        

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) # log softmax operation

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        if has_fewshot: # divide by the number of fewshot examples
            loss = loss.sum() / fewshot_ct
        else:
            loss = loss.view(anchor_count, batch_size).mean() # mean here is the 1/2N term

        return loss, fewshot_ct