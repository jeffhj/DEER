from typing import List, Dict
import json
import torch
from copy import deepcopy
import torch
from transformers import BertTokenizer

def my_mean(nums:List[int]):
    return len(nums)/sum([1/num for num in nums])

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_file:str,
                 global_rank=-1, 
                 world_size=-1):
        self.load_data(data_file, global_rank=global_rank, world_size=world_size)
        
    def load_data(self, data_file:str, global_rank=-1, world_size=-1):
        with open(data_file) as f_in:
            samples = json.load(f_in)
        
        data = []
        for k, sample in enumerate(samples):
            if global_rank > -1 and not k%world_size==global_rank:
                continue
            answer:str = sample['target']
            sources:List[str] = sample['source']
            entity:List[str] = sample['entity']
            triple:list = sample['triple']
            contexts = [[{'e1' : entity[tri['e1']], 
                          'e2' : entity[tri['e2']], 
                          'sent' : sources[tri['sent']],
                          'score' : tri['score']} for tri in path] for path in triple]
            
            data.append({'target_pair' : sample['pair'],
                         'target' : answer,
                         'ctxs' : contexts,
                         'index' : k})
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class FiDDataset(Dataset):
    def __init__(self, data_file: str, n_context: int, global_rank=-1, world_size=-1, no_sent=False,
                 no_path=False, duplicate_sample=True):
        super().__init__(data_file, global_rank, world_size)
        self.n_context = n_context
        self.no_sent = no_sent
        self.no_path = no_path
        self.duplicate_sample = duplicate_sample
        
    def __getitem__(self, index:int):
        example = self.data[index]
        question = 'entity1: {} entity2: {}'.format(*example['target_pair']).lower()
        target = example['target']
        contexts = []
        for ctx in example['ctxs'][:self.n_context]:
            path = [ctx[0]['e1']]
            sents = []
            for i, tri in enumerate(ctx):
                path.append(tri['e2'])
                sents.append('sentence%d: %s' % (i+1, tri['sent']))
            path = '; '.join(path)
            sents = ' '.join(sents)
            contexts.append('%s %s %s' % (question, 'path: ' + path if not self.no_path else '', sents if not self.no_sent else ''))
        
        if len(contexts) < self.n_context:
            print('should not happen')
            if self.duplicate_sample:
                while len(contexts) < self.n_context:
                    append_list = deepcopy(contexts[:self.n_context - len(contexts)])
                    contexts.extend(append_list)
            else:
                contexts.extend([question] * (self.n_context - len(contexts)))

        return {
            'index' : index,
            'target' : target,
            'passages' : contexts
        }


def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            padding='longest',
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids:torch.Tensor = target["input_ids"]
        target_mask:torch.Tensor = target["attention_mask"]
        target_mask = target_mask.bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        text_passages = [example['passages'] for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)

        return (index, target_ids, target_mask, passage_ids, passage_masks)
    
class DKGRECollator:
    def __init__(self, tokenizer:BertTokenizer, rid2label:Dict[str, int]):
        self.tokenizer = tokenizer
        self.rid2label = rid2label
        
    def __call__(self, batch):
        temp = []
        for sample in batch:
            subj_span, obj_span = sample['span']
            sent = sample['sentence']
            (first_split_start, first_split_end), (second_split_start, second_split_end) = (subj_span, obj_span) if subj_span[0] < obj_span[0] else (obj_span, subj_span)
            first_chunk = sent[:first_split_start]
            first_ent_chunk = sent[first_split_start:first_split_end]
            middle_chunk = sent[first_split_end:second_split_start]
            second_ent_chunk = sent[second_split_start:second_split_end]
            last_chunk = sent[second_split_end:]
            if subj_span[0] < obj_span[0]:
                marked_sent = ' '.join(first_chunk + ['<E1>'] + first_ent_chunk + ['</E1>'] + middle_chunk + ['<E2>'] + second_ent_chunk + ['</E2>'] + last_chunk)
            else:
                marked_sent = ' '.join(first_chunk + ['<E2>'] + first_ent_chunk + ['</E2>'] + middle_chunk + ['<E1>'] + second_ent_chunk + ['</E1>'] + last_chunk)
            temp.append(marked_sent)
        
        inputs = self.tokenizer.batch_encode_plus(
            temp,
            max_length=100,
            padding='longest',
            return_tensors='pt',
            truncation=True)
        labels = torch.LongTensor([self.rid2label[sample['rel_id'][0]] for sample in batch])
        sub_idx = inputs['input_ids'] == self.tokenizer.convert_tokens_to_ids('<E1>')
        obj_idx = inputs['input_ids'] == self.tokenizer.convert_tokens_to_ids('<E2>')
        return (inputs, labels, sub_idx, obj_idx)