from torch import nn

class VGG19(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG19, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.flat = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, mode='train'):
        x = self.pool1(self.relu(self.conv2(self.relu(self.conv1(x)))))
        x = self.pool2(self.relu(self.conv4(self.relu(self.conv3(x)))))
        x = self.pool3(self.relu(self.conv8(self.relu(self.conv7(self.relu(self.conv6(self.relu(self.conv5(x)))))))))
        x = self.pool4(self.relu(self.conv12(self.relu(self.conv11(self.relu(self.conv10(self.relu(self.conv9(x)))))))))
        x = self.pool5(self.relu(self.conv16(self.relu(self.conv15(self.relu(self.conv14(self.relu(self.conv13(x)))))))))
        x = self.flat(x)
        x = self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))
        x = self.softmax(x)
        if mode == 'train':
            return x
        return x.argmax(dim=1)