from transformers import AutoTokenizer, DistilBertModel
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

# Max Pooling - Take the max value over time for every dimension.
def max_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    return torch.max(token_embeddings, 1)[0]

def convert_text_to_feature(sentences, max_length=50):
    inputs = tokenizer.batch_encode_plus(
        sentences, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    sentence_embeddings = max_pooling(outputs, attention_mask)
    return sentence_embeddings

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
def save_checkpoint(model_dict, path):
    torch.save(model_dict, path)