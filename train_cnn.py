import torch.nn as nn
import img_cnn
import test_cnn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

MIN_EPOCH_NUMBER = 7
EARLY_STOP_PATIENCE = 2

def train(train_set: DataLoader, eval_set: DataLoader, test_set: DataLoader, max_epoch: int, models_path: str, log_file_name: str):
    print('train cnn - Start')
    log_file = open(log_file_name, "w", encoding="utf-8")

    model = img_cnn.ImgCNN()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {device}")
    model = model.to(device)
    
    print('set optimizer & loss')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_val_acc = 0
    best_val_epoch = -1
    best_test_acc = 0
    best_test_epoch = -1
    
    print('start training loops. #epochs = ' + str(max_epoch))
    print(f"{'Epoch':^7} | {'Train Loss':^12.2} | {'Train Acc':^10.2} | {'Test Acc':^10.2} | {'Eval Acc':^10.2} | {'Elapsed':^9}")
    print("-"*50)  
    
    log_file.write(f"{'Epoch':^7} | {'Train Loss':^12.2} | {'Train Acc':^10.2} | {'Test Acc':^10.2} | {'Eval Acc':^10.2} | {'Elapsed':^9}\n")
    log_file.write("-"*50 + "\n")
    log_file.flush()
        
    
    num_no_imp = 0
    for i in tqdm(range(max_epoch)):
        epoch = i + 1
        epoch_start_time = time.time()
        total_loss = 0
        num_batches = 0
        
        model.train()
        
        for data in train_set:
            images, labels = data
            
            optimizer.zero_grad()
            
            logits = model(images.to(device))
            
            loss = criterion(logits, labels.to(device))
            total_loss += loss.item()
            num_batches += 1
            loss.backward()
            optimizer.step()
            
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start_time
        
        # Validation test.
        val_acc = test_cnn.test(model, eval_set)
        train_acc = test_cnn.test(model, train_set)
        test_acc = test_cnn.test(model, test_set)
        val_acc *= 100
        train_acc *= 100
        test_acc *= 100
        print(f"{epoch:^7} | {avg_loss:^12.2f} | {train_acc:^10.2f} | {test_acc:^10.2f} |  {val_acc:^10.2f} | {epoch_time:^9f}")
        log_file.write(f"{epoch:^7} | {avg_loss:^12.2f} | {train_acc:^10.2f} | {test_acc:^10.2f} |  {val_acc:^10.2f} | {epoch_time:^9f}\n")
        log_file.flush()
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            num_no_imp = 0

            if epoch >= MIN_EPOCH_NUMBER:
                torch.save(model.state_dict(), models_path + f"{epoch}_{val_acc:.3f}.model")
        else:
            num_no_imp += 1
            
        if (num_no_imp > EARLY_STOP_PATIENCE) and (epoch >= MIN_EPOCH_NUMBER):
            print('early stop exit')
            log_file.write('\tEarly Stop exit\n')
            log_file.flush()
            break
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_epoch = epoch
    
    print('train_cnn_nlp - end')
    print("Best Val Acc = {:.2f}".format(best_val_acc) + ", best epoch = " + str(best_val_epoch))
    print("Best Test Acc = {:.2f}".format(best_test_acc) + ", best epoch = " + str(best_test_epoch))
    log_file.write("Best Val Acc = {:.2f}".format(best_val_acc) + ", best epoch = " + str(best_val_epoch) + "\n")
    log_file.write("Best Test Acc = {:.2f}".format(best_test_acc) + ", best epoch = " + str(best_test_epoch) + "\n")
    log_file.close()
