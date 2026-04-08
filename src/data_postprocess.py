import os
import re
import json
from tqdm import tqdm
from langchain.prompts import PromptTemplate


class DataPostprocess:
    """
    the postprocess of generated data
    """
    def __init__(
            self, 
            data_path: str,
            ablation: str=None,
            threshold: float=7.0,
        ) -> None:
        self.data_path = data_path
        self.ablation = ablation
        self.threshold = threshold
        self.sft_output_path = os.path.join(data_path.split('.')[0], 'sft_output.json')
        self.dpo_output_path = os.path.join(data_path.split('.')[0], 'dpo_output.json')
        self.next_seed_path = os.path.join(data_path.split('.')[0], 'seed.json')


        with open(self.data_path, 'r', encoding='utf-8') as json_file:
            self.data = json.load(json_file)
    
    def __call__(self):
        if self.ablation == "sft":
            self.sft_postprocess()
        elif self.ablation == "dpo":
            self.dpo_postprocess()
        elif self.ablation is None:
            self.all_postprocess()
        else:
            raise ValueError(f"Unknown ablation mode: {self.ablation}")
    
    def sft_postprocess(self):
        new_data = []
        for data in tqdm(self.data, desc="SFT Postprocessing"):
            if(data['score'] <= self.threshold):
                for new_prompt,idx in enumerate(data['new_prompt']):
                    for new_response,rid in enumerate(data['new_response'][idx]):
                        for signal_response in new_response:
                            data_item = {
                                'prompt': new_prompt,
                                'response': signal_response,
                                'review': data['new_review'][idx][rid],
                                'score': data['new_score'][idx][rid]
                            }
                            new_data.append(data_item)
        with open(self.sft_output_path, 'w', encoding='utf-8') as json_file:
            json.dump(new_data, json_file, indent=4, ensure_ascii=False)
    
    def dpo_postprocess(self):
        new_data = []
        for data in tqdm(self.data, desc="DPO Postprocessing"):
            if(data['score'] > self.threshold):
                for new_response,idx in enumerate(data['new_response'][0]):
                    for signal_response in new_response:
                        data_item = {
                            'prompt': data['prompt'],
                            'good_response':data['response'],
                            'good_response_review':data['review'],
                            'good_response_score':data['score'],
                            'bad_response':signal_response,
                            'bad_response_review':data['new_review'][0][idx],
                            'bad_response_score':data['new_score'][0][idx]
                        }
                        new_data.append(data_item)
        with open(self.dpo_output_path, 'w', encoding='utf-8') as json_file:
            json.dump(new_data, json_file, indent=4, ensure_ascii=False)
    
    def all_postprocess(self):
        sft_new_data = []
        dpo_new_data = []
        for data in tqdm(self.data, desc="All Postprocessing"):
            if(data['score'] <= self.threshold):
                for new_prompt,idx in enumerate(data['new_prompt']):
                    for new_response,rid in enumerate(data['new_response'][idx]):
                        for signal_response in new_response:
                            data_item = {
                                'prompt': new_prompt,
                                'response': signal_response,
                                'review': data['new_review'][idx][rid],
                                'score': data['new_score'][idx][rid]
                            }
                            sft_new_data.append(data_item)
            elif(data['score'] > self.threshold):
                for new_response,idx in enumerate(data['new_response'][0]):
                    for signal_response in new_response:
                        data_item = {
                            'prompt': data['prompt'],
                            'good_response':data['response'],
                            'good_response_review':data['review'],
                            'good_response_score':data['score'],
                            'bad_response':signal_response,
                            'bad_response_review':data['new_review'][0][idx],
                            'bad_response_score':data['new_score'][0][idx]
                        }
                        dpo_new_data.append(data_item)
        with open(self.sft_output_path, 'w', encoding='utf-8') as json_file:
            json.dump(sft_new_data, json_file, indent=4, ensure_ascii=False)
        with open(self.dpo_output_path, 'w', encoding='utf-8') as json_file:
            json.dump(dpo_new_data, json_file, indent=4, ensure_ascii=False)
        with open(self.next_seed_path, 'w', encoding='utf-8') as json_file:
            json.dump(sft_new_data, json_file, indent=4, ensure_ascii=False)