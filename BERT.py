import warnings
import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
warnings.filterwarnings("ignore")
matplotlib.style.use("classic")
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
import os
import random
import numpy as np
from transformers import WEIGHTS_NAME, CONFIG_NAME
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import time
import datetime


if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


final_stock_list = ['600000', '600009', '600016', '600031', '600036', '600048', '600309', '600547', '600570', '600585', 
                    '600588', '600690', '600703', '600745', '600809', '600887', '600893', '601088', '601138', '601288', 
                    '601336', '601398', '601601', '601688', '601818', '601857', '601888', '603259', '603288', '603501', '603986']
final_stock_name = ["浦发银行", "上海机场", "民生银行", "三一重工", "招商银行", "保利地产", "万华化学", "山东黄金", "恒生电子", "海螺水泥",
                    "用友网络", "海尔智家", "三安光电", "闻泰科技", "山西汾酒", "伊利股份", "航发动力", "中国神华", "工业富联", "农业银行", 
                    "新华保险", "工商银行", "中国太保", "华泰证券", "光大银行", "中国石油", "中国国旅", "药明康德", "海天味业", "韦尔股份", "兆易创新"]

total_data_tv = pd.DataFrame(columns = ["Content", "Tag"])

# remove company name from the new data
for i in tqdm(range(len(final_stock_list))):
    train_data = pd.read_csv(f"data_cleaned/Train/{final_stock_list[i]}NLP_Train_cleaned.csv")
    valid_data = pd.read_csv(f"data_cleaned/Valid/{final_stock_list[i]}NLP_Valid_cleaned.csv")
    for k in range(len(train_data)):
        content = train_data["Content"].loc[k]
        tag = train_data["Tag"].loc[k]
        if tag == -2:
            train_data["Tag"].loc[k] = 0
        elif tag == -1:
            train_data["Tag"].loc[k] = 1
        elif tag == 0:
            train_data["Tag"].loc[k] = 2
        elif tag == 1:
            train_data["Tag"].loc[k] = 3
        else:
            train_data["Tag"].loc[k] = 4
        content = content.replace(final_stock_name[i], "")
        train_data["Content"].loc[k] = content
    for j in range(len(valid_data)):
        content = valid_data["Content"].loc[j]
        tag = valid_data["Tag"].loc[j]
        if tag == -2:
            valid_data["Tag"].loc[j] = 0
        elif tag == -1:
            valid_data["Tag"].loc[j] = 1
        elif tag == 0:
            valid_data["Tag"].loc[j] = 2
        elif tag == 1:
            valid_data["Tag"].loc[j] = 3
        else:
            valid_data["Tag"].loc[j] = 4
        content = content.replace(final_stock_name[i], "")
        valid_data["Content"].loc[j] = content
    total_data_tv = total_data_tv.append(train_data[["Content", "Tag"]])
    total_data_tv = total_data_tv.append(valid_data[["Content", "Tag"]])
    
total_data_tv = total_data_tv.reset_index(drop=True)

# get sentence and label
sentences = total_data_tv.Content.values
labels = total_data_tv.Tag.values

# import pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

print(' Raw Sentence: ', sentences[0])
print('Tokenized Sentence: ', tokenizer.tokenize(sentences[0]))
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

max_len = 0
lengthOfsentence = []
# loop for every sentence
for sent in sentences:

    lengthOfsentence.append(len(sent))
    # find the max length of sentence
    max_len = max(max_len, len(sent))

print('The max length of sentences is: ', max_len)
print(type(sentences))
print(sentences.shape)

plt.plot(lengthOfsentence)
plt.ylabel('some numbers')
plt.show()

input_ids = []
attention_masks = []

for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 50,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # put encoded sentence into list.
    input_ids.append(encoded_dict['input_ids'])
    
    # add attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

print(labels)

# convert lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = np.int_(labels)
labels = labels.astype("int64")

labels = torch.tensor(labels)
print(labels.shape)

print('Original Sentence: ', sentences[0])
print('Token IDs:', input_ids[0])
print('Attention_masks:', attention_masks[0])


# convert inputs into TensorDataset。
dataset = TensorDataset(input_ids, attention_masks, labels)

# calculate train_size and val_size lengths.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# 90% of the dataset is train_dataset, 10% of the dataset is val_dataset.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print('{:>5,} training data'.format(train_size))
print('{:>5,} validation data'.format(val_size))


batch_size = 32

# design DataLoaders for train data and validation data.
train_dataloader = DataLoader(
            train_dataset,  # train data.
            sampler = RandomSampler(train_dataset), # random sequential
            batch_size = batch_size 
        )

validation_dataloader = DataLoader(
            val_dataset, # val_data.
            sampler = RandomSampler(val_dataset), # random sequential
            batch_size = batch_size 
        )

model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese", # use 12-layer BERT model.
    num_labels = 5, # set number of classification label to be 5.
    output_attentions = False, # do not return attentions weights.
    output_hidden_states = False, # do not return all hidden-states.
)
model.cuda()


# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


# AdamW is a class from huggingface library, 'W' is the meaning of 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate
                  eps = 1e-8 # args.adam_epsilon
                )

# bert's recommended epoch number is 2-4
epochs = 3

# training steps number: [number of batches] x [number of epochs].
total_steps = len(train_dataloader) * epochs

# design learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):    
    elapsed_rounded = int(round((elapsed)))    
    # return hh:mm:ss formate time
    return str(datetime.timedelta(seconds=elapsed_rounded))


output_dir = "models/"
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

# set seed
seed_val = 615
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# record training ,validation loss ,validation accuracy and timings.
training_stats = []

# set total time
total_t0 = time.time()
best_val_accuracy = 0

for epoch_i in range(0, epochs):      
    print('Epoch {:} / {:}'.format(epoch_i + 1, epochs))  

    # record the time used in each epoch
    t0 = time.time()
    total_train_loss = 0
    total_train_accuracy = 0
    model.train()
  
    for step, batch in enumerate(train_dataloader):

        # output used time after each 40 batchs
        if step % 40 == 0 and not step == 0:            
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # `batch` contains three tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()        

        # forward
        loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)[:2]
       
        total_train_loss += loss.item()
        # backward upgrade gradients.
        loss.backward()
        # set the gradient bigger than 1.0 and set it to 1.0, to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update model parameters
        optimizer.step()
        # update learning rate.
        scheduler.step()        
             
        logit = logits.detach().cpu().numpy()
        label_id = b_labels.to('cpu').numpy()
        # calculate training accuracy
        total_train_accuracy += flat_accuracy(logit, label_id)    
     
    # calculate batch average loss
    avg_train_loss = total_train_loss / len(train_dataloader)      
    # calculate training time
    training_time = format_time(time.time() - t0)
    
    # accuracy on the train_data
    avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    print("  Tain acc: {0:.2f}".format(avg_train_accuracy))
    print("  Avg train loss loss: {0:.2f}".format(avg_train_loss))
    print("  training time: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================

    t0 = time.time()

    # set model to be valuation mode，in this mode, dropout layers and dropout rate will be different
    model.eval()

    # set parameters
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        # `batch` has tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():
            (loss, logits) = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)[:2]
            
        # Calculate validation loss.
        total_eval_loss += loss.item()        
        logit = logits.detach().cpu().numpy()
        label_id = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logit, label_id)
        
    # calculate validation accuracy.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("")
    print("  Validation accuracy: {0:.2f}".format(avg_val_accuracy))
    
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        torch.save(model.state_dict(),output_model_file)
        model.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(output_dir)
         

    # calculate batches's average loss
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # calculate validation time.
    validation_time = format_time(time.time() - t0)
    
    print("  Avg Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Test time {:}".format(validation_time))

    # record model parameters
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("total training time {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
