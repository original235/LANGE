import os
import re
import json
from tqdm import tqdm
from openai import OpenAI
from langchain.prompts import PromptTemplate

import prompt_template

class DataSource:
    """
    Generate data based on data quality using LLM API
    """
    def __init__(
            self, 
            model_name: str, 
            api_key: str,
            base_url: str,
            data_path: str,
            output_path: str,
            threshold: float=7.0,
            api_num_worker: int = 4,
        ) -> None:
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.api_num_worker = api_num_worker
        
        ## 2.read data
        with open(data_path, 'r', encoding='utf-8') as json_file:
            self.origin_data = json.load(json_file)
        self.skip_len = 0
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as json_file:
                reviewed_data = json.load(json_file)
            self.skip_len = len(reviewed_data)
            self.origin_data[:self.skip_len] = reviewed_data
            print(f'{self.skip_len} pieces of data were skipped') 
        self.output_path = output_path
        self.threshold = threshold

    def sample(
            self, 
            prompt: str, 
            N: int=4,
            max_new_tokens: int=512,
            top_p: float=0.9,
            temperature: float=0.7
        ) -> list[str]:
        '''
        Sample a batch response
        '''
        response_batch = []
        for _ in range(N):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {'role': 'system', 'content': prompt_template.SYS_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                response_batch.append(response.choices[0].message.content)
            except Exception as e:
                print(f"API Error: {e}")
                response_batch.append("")
        return response_batch
    
    def final_answer(self, batch_data) -> None:
        pattern = r'Final Answer:([\s\S]*)'
        for i in reversed(range(len(batch_data))):
            match = re.search(pattern, batch_data[i])
            if match:
                batch_data[i] = match.group(1).strip()
            else:
                del batch_data[i]
    
    def save(self, idx) -> None:
        with open(self.output_path, 'w', encoding='utf-8') as outfile:
            json.dump(self.origin_data[:idx+1], outfile, indent=4, ensure_ascii=False)
    
    def process_item(self, data, ablation):
        if ablation is None:
            if data['score'] > self.threshold:
                pt = PromptTemplate.from_template(template=prompt_template.LANGGPT4ResNShot)
                prompt = pt.format(instruction=data['prompt'], response=data['response'])
                res_batch = self.sample(prompt=prompt)
                self.final_answer(res_batch)
                data['new_prompt'] = [data['prompt']]
                data['new_response'] = [res_batch]
            else:
                pt = PromptTemplate.from_template(template=prompt_template.LANGGPT4NewInstructionNShot)
                prompt = pt.format(instruction=data['prompt'], response=data['response'])
                pro_batch = self.sample(prompt=prompt)
                self.final_answer(pro_batch)
                data['new_prompt'] = pro_batch
                new_res = []
                for pro in pro_batch:
                    res_batch = self.sample(prompt=pro)
                    new_res.append(res_batch)
                data['new_response'] = new_res
        elif ablation == "sft":
            if data['score'] <= self.threshold:
                pt = PromptTemplate.from_template(template=prompt_template.LANGGPT4NewInstructionNShot)
                prompt = pt.format(instruction=data['prompt'], response=data['response'])
                pro_batch = self.sample(prompt=prompt)
                self.final_answer(pro_batch)
                data['new_prompt'] = pro_batch
                new_res = []
                for pro in pro_batch:
                    res_batch = self.sample(prompt=pro)
                    new_res.append(res_batch)
                data['new_response'] = new_res
        elif ablation == "dpo":
            if data['score'] > self.threshold:
                pt = PromptTemplate.from_template(template=prompt_template.LANGGPT4ResNShot)
                prompt = pt.format(instruction=data['prompt'], response=data['response'])
                res_batch = self.sample(prompt=prompt)
                self.final_answer(res_batch)
                data['new_prompt'] = [data['prompt']]
                data['new_response'] = [res_batch]
        else:
            raise ValueError(f"Unknown ablation mode: {ablation}")
        return data

    def __call__(
        self, 
        ablation: str=None  # Is it ablation? "sft", "dpo" or None
    ):
        import concurrent.futures
        items_to_process = self.origin_data[self.skip_len:]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.api_num_worker) as executor:
            results = executor.map(lambda d: self.process_item(d, ablation), items_to_process)
            
            for i, processed_data in enumerate(tqdm(results, total=len(items_to_process))):
                self.origin_data[self.skip_len + i] = processed_data
                if i % 10 == 0:
                    self.save(self.skip_len + i)
                    
        self.save(len(self.origin_data) - 1)
