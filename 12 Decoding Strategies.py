# import os
# os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
from transformers import GPT2LMHeadModel,GPT2Tokenizer,set_seed
import torch
import torch.nn.functional as F
import numpy as np

cache_dir='./transformers/gpt2'
model=GPT2LMHeadModel.from_pretrained(cache_dir)
tokenizer=GPT2Tokenizer.from_pretrained(cache_dir)

# np.random.seed(1)
# print("Number of Tokens in Dictionary = %d"%(tokenizer.vocab_size))
# for i in range(20):
#     index=np.random.randint(tokenizer.vocab_size)
#     print("Token: %d"%(index)+tokenizer.decode(torch.tensor(index),skip_special_tokens=True))

def SampleNextToken(input_tokens,model,tokenizer):
    outputs=model(input_ids=input_tokens['input_ids'],attention_mask=input_tokens['attention_mask'])
    prob_over_tokens=F.softmax(outputs.logits,dim=-1).detach().numpy()[0,-1]
    next_token=np.random.choice(len(prob_over_tokens),p=prob_over_tokens)
    next_token=np.array([next_token])
    output_tokens=input_tokens
    output_tokens["input_ids"]=torch.cat((output_tokens['input_ids'],torch.tensor([next_token])),dim=1)
    output_tokens['attention_mask']=torch.cat((output_tokens['attention_mask'],torch.tensor([[1]])),dim=1)
    output_tokens['last_token_prob']=prob_over_tokens[next_token]
    return output_tokens

# set_seed(0)
# input_txt="The best thing is"
# input_tokens=tokenizer(input_txt,return_tensors='pt')
# for i in range(20):
#     input_tokens=SampleNextToken(input_tokens,model,tokenizer)
#     print(tokenizer.decode(input_tokens["input_ids"][0],skip_special_tokens=True))

def GetBestNextToken(input_tokens,model,tokenizer):
    outputs=model(input_ids=input_tokens['input_ids'],attention_mask=input_tokens['attention_mask'])
    prob_over_tokens=F.softmax(outputs.logits,dim=-1).detach().numpy()[0,-1]
    next_token=[np.argmax(prob_over_tokens)]
    output_tokens=input_tokens
    output_tokens["input_ids"]=torch.cat((output_tokens['input_ids'],torch.tensor([next_token])),dim=1)
    output_tokens['attention_mask']=torch.cat((output_tokens['attention_mask'],torch.tensor([[1]])),dim=1)
    output_tokens['last_token_prob']=prob_over_tokens[next_token]
    return output_tokens

# set_seed(0)
# input_txt="The best thing is"
# input_tokens=tokenizer(input_txt,return_tensors='pt')
# for i in range(20):
#     input_tokens=GetBestNextToken(input_tokens,model,tokenizer)
#     print(tokenizer.decode(input_tokens["input_ids"][0],skip_special_tokens=True))

def TopKSampling(input_tokens,model,tokenizer,k=20):
    outputs=model(input_ids=input_tokens['input_ids'],attention_mask=input_tokens['attention_mask'])
    prob_over_tokens=F.softmax(outputs.logits,dim=-1).detach().numpy()[0,-1]
    # sorted_indices = np.argsort(prob_over_tokens)[::-1]  # Get indices in reverse order
    # sorted_prob_over_tokens = prob_over_tokens[sorted_indices]  # Sort the probabilities
    # top_k_indices = sorted_indices[:k]  # Get the top K indices
    # sorted_prob_over_tokens = sorted_prob_over_tokens[:k]  # Get the top K probabilities
    sorted_prob_over_tokens=np.sort(prob_over_tokens)
    kth_prob_value=sorted_prob_over_tokens[k-1]
    prob_over_tokens[prob_over_tokens<kth_prob_value]=0
    prob_over_tokens=prob_over_tokens/prob_over_tokens.sum()
    next_token=np.random.choice(len(prob_over_tokens),1,replace=False,p=prob_over_tokens)
    output_tokens=input_tokens
    output_tokens["input_ids"]=torch.cat((output_tokens['input_ids'],torch.tensor([next_token])),dim=1)
    output_tokens['attention_mask']=torch.cat((output_tokens['attention_mask'],torch.tensor([[1]])),dim=1)
    output_tokens['last_token_prob']=prob_over_tokens[next_token]
    return output_tokens

# set_seed(0)
# input_txt="The best thing is"
# input_tokens=tokenizer(input_txt,return_tensors='pt')
# for i in range(20):
#     input_tokens=TopKSampling(input_tokens,model,tokenizer)
#     print(tokenizer.decode(input_tokens["input_ids"][0],skip_special_tokens=True))

def NucleusSamplling(input_tokens,model,tokenizer,thresh=0.25):
    outputs=model(input_ids=input_tokens['input_ids'],attention_mask=input_tokens['attention_mask'])
    prob_over_tokens=F.softmax(outputs.logits,dim=-1).detach().numpy()[0,-1]
    sorted_probs_decreasing=np.sort(prob_over_tokens)
    cum_sum_probs=np.cumsum(sorted_probs_decreasing)
    thresh_index=np.argmax(cum_sum_probs>thresh)
    # print("Choosing From %d tokens"%(thresh_index))
    thresh_prob=prob_over_tokens[thresh_index]
    prob_over_tokens[prob_over_tokens<thresh_prob]=0
    prob_over_tokens=prob_over_tokens/np.sum(prob_over_tokens)
    next_token=np.random.choice(len(prob_over_tokens),1,replace=False,p=prob_over_tokens)
    output_tokens=input_tokens
    output_tokens["input_ids"]=torch.cat((output_tokens['input_ids'],torch.tensor([next_token])),dim=1)
    output_tokens['attention_mask']=torch.cat((output_tokens['attention_mask'],torch.tensor([[1]])),dim=1)
    output_tokens['last_token_prob']=prob_over_tokens[next_token]
    return output_tokens

# set_seed(0)
# input_txt="The best thing is"
# input_tokens=tokenizer(input_txt,return_tensors='pt')
# for i in range(20):
#     input_tokens=NucleusSamplling(input_tokens,model,tokenizer)
#     print(tokenizer.decode(input_tokens["input_ids"][0],skip_special_tokens=True))

def GetKthMostLikelyToken(input_tokens,model,tokenizer,k):
    outputs=model(input_ids=input_tokens['input_ids'],attention_mask=input_tokens['attention_mask'])
    prob_over_tokens=F.softmax(outputs.logits,dim=-1).detach().numpy()[0,-1]
    sorted_indices=np.argsort(prob_over_tokens)[::-1]
    sorted_prob_over_tokens=prob_over_tokens[sorted_indices]
    kth_prob_value=sorted_prob_over_tokens[k]
    next_token=np.where(prob_over_tokens==kth_prob_value)[0]
    output_tokens=input_tokens
    output_tokens["input_ids"]=torch.cat((output_tokens['input_ids'],torch.tensor([next_token])),dim=1)
    output_tokens['attention_mask']=torch.cat((output_tokens['attention_mask'],torch.tensor([[1]])),dim=1)
    output_tokens['last_token_prob']=prob_over_tokens[next_token]
    output_tokens['log_prob']=output_tokens['log_prob']+np.log(prob_over_tokens[next_token])
    return output_tokens

# set_seed(0)
# input_txt="The best thing is"
# input_tokens=tokenizer(input_txt,return_tensors='pt')
# input_tokens['log_prob']=0.0
# for i in range(20):
#     input_tokens=GetKthMostLikelyToken(input_tokens,model,tokenizer,k=1)
#     print(tokenizer.decode(input_tokens["input_ids"][0],skip_special_tokens=True))

def PrintBeams(beams):
    for index,beam in enumerate(beams):
        print("Beam %d, Prob %3.3f: "%(index,beam['log_prob'])+tokenizer.decode(beam["input_ids"][0],skip_special_tokens=True))
    print('---')

def DoBeamSearch(input_tokens_in,model,tokenizer,n_beam=7,beam_length=10):
    input_tokens['log_prob']=0.0
    beams=[None]*n_beam
    for c_k in range(n_beam):
        beams[c_k]=dict(input_tokens_in)
        beams[c_k]=GetKthMostLikelyToken(beams[c_k],model,tokenizer,c_k)
    PrintBeams(beams)
    for c_pos in range(beam_length-1):
        beams_all=[None]*(n_beam*n_beam)
        log_probs_all=np.zeros(n_beam*n_beam)
        for c_beam in range(n_beam):
            for c_k in range(n_beam):
                beams_all[c_beam*n_beam+c_k]=dict(GetKthMostLikelyToken(beams[c_beam],model,tokenizer,c_k))
                log_probs_all[c_beam*n_beam+c_k]=beams_all[c_beam*n_beam+c_k]['log_prob']
        sorted_index=np.argsort(np.array(log_probs_all)*-1)
        for c_k in range(n_beam):
            beams[c_k]=dict(beams_all[sorted_index[c_k]])
        PrintBeams(beams)
    return beams[0]

set_seed(0)
input_txt="You should know that"
input_tokens=tokenizer(input_txt,return_tensors='pt')
n_beams=10
best_beam=DoBeamSearch(input_tokens,model,tokenizer)
print("Beam Search Result: ")
print(tokenizer.decode(best_beam["input_ids"][0],skip_special_tokens=True))
