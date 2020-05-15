import torch 
from torchvision import transforms , models 
from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np
import imageio
'''
Downloading: "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
to /home/yash/.cache/torch/checkpoints/vgg19-dcbb9e9d.pth
'''
device = ("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

model = models.vgg16(pretrained=True).features
for p in model.parameters():
    p.requires_grad = False
model.to(device)

def model_activations(input,model):
    layers = {
    '0' : 'conv1_1',
    '5' : 'conv2_1',
    '10': 'conv3_1',
    '19': 'conv4_1',
    '21': 'conv4_2',
    '28': 'conv5_1'
    }
    features = {}
    x = input
    x = x.unsqueeze(0)
    for name,layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x 
    
    return features

transform = transforms.Compose([transforms.Resize(300),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


content_filename = '/home/yash/Downloads/images/lana.jpg'
style_filename = '/home/yash/Downloads/images/style1.jpg'
content = Image.open(content_filename).convert("RGB")
content = transform(content).to(device)
print("Content shape => ", content.shape)
style = Image.open(style_filename).convert("RGB")
style = transform(style).to(device)

def imcnvt(image):
    x = image.to("cpu").clone().detach().numpy().squeeze()
    x = x.transpose(1,2,0)
    x = x*np.array((0.5,0.5,0.5)) + np.array((0.5,0.5,0.5))
    return x

fig, (ax1,ax2) = plt.subplots(1,2)

ax1.imshow(imcnvt(content),label = "Content")
ax2.imshow(imcnvt(style),label = "Style")
plt.show()

def gram_matrix(imgfeature):
    _,d,h,w = imgfeature.size()
    imgfeature = imgfeature.view(d,h*w)
    gram_mat = torch.mm(imgfeature,imgfeature.t())
    return gram_mat

target = content.clone().requires_grad_(True).to(device)

#set device to cuda if available
print("device = ",device)


style_features = model_activations(style,model)
content_features = model_activations(content,model)

style_wt_meas = {"conv1_1" : 1.0, 
                 "conv2_1" : 0.8,
                 "conv3_1" : 0.4,
                 "conv4_1" : 0.2,
                 "conv5_1" : 0.1}

style_grams = {layer:gram_matrix(style_features[layer]) for layer in style_features}

# hyperparameters
# more content_wt --> more of content img, more style_wt --> more of style img

# content_wt = 100
# style_wt = 1e8 # 10^8

content_wt = 300
style_wt = 1e10 

print_after = 200
epochs = 4000
optimizer = torch.optim.Adam([target],lr=0.007)

save_path = "neural_style_transfer_result/"

for i in range(1,epochs+1):
    target_features = model_activations(target,model)
    content_loss = torch.mean((content_features['conv4_2']-target_features['conv4_2'])**2)

    style_loss = 0
    for layer in style_wt_meas:
        style_gram = style_grams[layer]
        target_gram = target_features[layer]
        _,d,w,h = target_gram.shape
        target_gram = gram_matrix(target_gram)

        style_loss += (style_wt_meas[layer]*torch.mean((target_gram-style_gram)**2))/d*w*h
    
    total_loss = content_wt*content_loss + style_wt*style_loss 
                  # alpha                    beta
    
    print("Epoch {0}/{1}, Loss: {2}".format(i, epochs, total_loss))
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    filename = save_path+'stylized_at_iteration_%d.png' % i
    if i%print_after == 0:
        imageio.imwrite(filename, imcnvt(target))