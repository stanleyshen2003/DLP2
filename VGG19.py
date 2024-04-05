from torch import nn

class VGG19(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG19, self).__init__()
        
        self.relu = nn.nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.norm4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.norm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.norm6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.norm7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.norm8 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.norm9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.norm10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.norm11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.norm12 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.norm13 = nn.BatchNorm2d(512)
        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.norm14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.norm15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.norm16 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.flat = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, mode='train'):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.pool1(x)

        x = self.relu(self.norm3(self.conv3(x)))
        x = self.relu(self.norm4(self.conv4(x)))
        x = self.pool2(x)

        x = self.relu(self.norm5(self.conv5(x)))
        x = self.relu(self.norm6(self.conv6(x)))
        x = self.relu(self.norm7(self.conv7(x)))
        x = self.relu(self.norm8(self.conv8(x)))
        x = self.pool3(x)

        x = self.relu(self.norm9(self.conv9(x)))
        x = self.relu(self.norm10(self.conv10(x)))
        x = self.relu(self.norm11(self.conv11(x)))
        x = self.relu(self.norm12(self.conv12(x)))
        x = self.pool4(x)

        x = self.relu(self.norm13(self.conv13(x)))
        x = self.relu(self.norm14(self.conv14(x)))
        x = self.relu(self.norm15(self.conv15(x)))
        x = self.relu(self.norm16(self.conv16(x)))
        x = self.pool5(x)

        x = self.flat(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        if mode == 'train':
            return x
        return x.argmax(dim=1)
    
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)