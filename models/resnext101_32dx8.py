import torchvision.models as models

#Mainly using the pre-trianed models which were provided by pytorch with ResNext101_32dx8
# as a backup choice resnext50 was also under consideration#

def _return_resnext101 ():
    return models.resnext101_32x8d(pretrained=True)

def _return_resnext50 ():
    return models.resnext50_32x4d(pretrained=True)