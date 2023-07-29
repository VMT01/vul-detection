import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

###########################################################################################

VUL_DIR = '/home/bkcs/smart-contract-vulnerability-detect-project/Source Code/Authentication through tx.origin.csv'
NO_VUL_DIR = '/home/bkcs/smart-contract-vulnerability-detect-project/Source Code/Contracts_No_Vul.csv'
VUL_RATIO = 1.0
NO_VUL_RATIO = 1.0
SPLIT_RATIO = 0.8 # Train vs Val data
TITLE = 'Authentication through tx.origin'
EPOCHS = 10
BATCH_SIZE = 128
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
LEARNING_RATE = 1e-05

tokenizer = RobertaTokenizer.from_pretrained('jackaduma/SecRoBERTa', do_lower_case=True)
encoder = RobertaForSequenceClassification.from_pretrained("jackaduma/SecRoBERTa", num_labels = 2)
optimizer = torch.optim.Adam(params=encoder.parameters(), lr=LEARNING_RATE)

###########################################################################################

class Discriminator(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.X = X
        self.targets = y
        self.max_len = max_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        text = str(self.X[index])

        inputs = self.tokenizer(
            text,
            None,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }

###########################################################################################

def read_data(vul_ratio=0.1, no_vul_ratio=0.1):
    df_vul = pd.read_csv(VUL_DIR, usecols=['BYTECODE']).dropna().drop_duplicates(subset=['BYTECODE']).sample(frac=vul_ratio)
    df_vul['LABEL'] = 1

    df_no_vul = pd.read_csv(NO_VUL_DIR, usecols=['OPCODE']).dropna().drop_duplicates(subset=['OPCODE']).sample(frac=no_vul_ratio)
    df_no_vul.rename(columns={'OPCODE':'BYTECODE'}, inplace=True)
    df_no_vul['LABEL'] = 0

    df = pd.concat([df_no_vul, df_vul]).sample(frac=1)

    X = df['BYTECODE'].values
    y = df['LABEL'].values

    return X, y

def prepare_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


    training_set = Discriminator(X_train, y_train, tokenizer, max_len=512)
    validating_set = Discriminator(X_val, y_val, tokenizer, max_len=512)
    testing_set = Discriminator(X_test, y_test, tokenizer, max_len=512)

    return training_set, validating_set, testing_set, y_test

def calculate_score(y_true, preds):
    F1_score = f1_score(y_true, preds, average='macro')
    acc_score = accuracy_score(y_true, preds)

    return acc_score, F1_score

def train_steps(train_loader, model, loss_function, optimizer):
    print('Training...')
    training_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    train_acc = 0.
    train_f1 = 0.

    model.train()

    for step, batch in enumerate(train_loader):
        ids = batch['ids'] #.to(device)
        mask = batch['mask'] #.to(device)
        token_type_ids = batch['token_type_ids'] #.to(device)
        targets = batch['targets'] #.to(device)
        
        outputs = model(ids, attention_mask=mask, token_type_ids=token_type_ids)
        logits = outputs.logits

        loss = loss_function(logits, targets)
        training_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)
        label_ids = targets.to('cpu').numpy()

        acc_score, F1_score = calculate_score(label_ids, preds)
        train_acc += acc_score
        train_f1 += F1_score
        nb_tr_steps += 1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # # When using GPU
        optimizer.step()

    epoch_loss = training_loss / nb_tr_steps
    epoch_acc = train_acc / nb_tr_steps
    epoch_f1 = train_f1 / nb_tr_steps
    print(" Accuracy: ", epoch_acc)
    print(" F1 score: ", epoch_f1)
    print(" Average training loss: ", epoch_loss)
    return epoch_loss, epoch_acc, epoch_f1

def evaluate_steps(validating_loader, model, loss_function):
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []
    total_labels = []
    # iterate over batches
    for step, batch in enumerate(validating_loader):
        # push the batch to gpu
        b_input_ids = batch['ids'] #.to(device)
        b_input_mask = batch['mask'] #.to(device)
        token_type_ids = batch['token_type_ids'] #.to(device)
        b_labels = batch['targets'] #.to(device)

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            output = model(b_input_ids, attention_mask=b_input_mask, token_type_ids=token_type_ids)
            logits = output.logits

            # compute the validation loss between actual and predicted values
            loss = loss_function(logits, b_labels)

            total_loss = total_loss + loss.item()

            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)
            total_preds += list(preds)
            total_labels += b_labels.tolist()
    # compute the validation loss of the epoch
    avg_loss = total_loss / len(validating_loader)
    acc_score, F1_score = calculate_score(total_labels, total_preds)

    print(" Accuracy: ", acc_score)
    print(" F1 score: ", F1_score)
    print(" Average training loss: ", avg_loss)
    return avg_loss, acc_score, F1_score

def predict(testing_loader, model):
    print('\nTesting...')
    model.eval()

    total_preds = []
    total_labels = []

    for step, batch in enumerate(testing_loader):
        b_input_ids = batch['ids'] # .to(device)
        b_input_mask = batch['mask'] # .to(device)
        token_type_ids = batch['token_type_ids'] # .to(device)
        b_labels = batch['targets'] # .to(device)

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            output = model(b_input_ids, attention_mask=b_input_mask, token_type_ids=token_type_ids)
            logits = output.logits

            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)
            total_preds += list(preds)
            total_labels += b_labels.tolist()

    return total_preds

def train(model, train_loader, valid_loader, optimizer):
    loss_function = torch.nn.CrossEntropyLoss()
    best_valid_loss = float('inf')

    train_losses, valid_losses = [], []

    for epoch in range(EPOCHS):
        print('\n Epoch {}/{}'.format(epoch + 1, EPOCHS))
        train_loss, train_acc, train_f1 = train_steps(train_loader, encoder, loss_function, optimizer)
        val_loss, val_acc, val_f1 = evaluate_steps(valid_loader, encoder, loss_function)

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss 
            torch.save(encoder.state_dict(), './secrobert/{}.pt'.format(TITLE))

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(val_loss)


    return train_losses, valid_losses

if __name__ == "__main__":
	X, y = read_data(vul_ratio=VUL_RATIO, no_vul_ratio=NO_VUL_RATIO)
	training_set, validating_set, testing_set, y_test = prepare_data(X, y)

	train_params = {
	    'batch_size': TRAIN_BATCH_SIZE,
	    'shuffle': True,
	    'num_workers': 0
	}

	val_params = {
	    'batch_size': VALID_BATCH_SIZE,
	    'shuffle': True,
	    'num_workers': 0
	}

	test_params = {
	    'batch_size': VALID_BATCH_SIZE,
	    'shuffle': False,
	    'num_workers': 0
	}

	train_loader = DataLoader(training_set, **train_params)
	valid_loader = DataLoader(validating_set, **val_params)
	test_loader = DataLoader(testing_set, **test_params)

	train(encoder, train_loader, valid_loader, optimizer)

	encoder.load_state_dict(torch.load('./secrobert/{}.pt'.format(TITLE)))

	y_pred = predict(test_loader, encoder)
	y_pred = np.array(y_pred, dtype=int)

	result = classification_report(y_test.astype(int), y_pred.astype(int), output_dict=True)
	pd.DataFrame(result).transpose().to_csv('./secrobert/{}.csv'.format(TITLE))

