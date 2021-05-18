import streamlit as st
import numpy as np
import os

from nbdt.model import SoftNBDT
from nbdt.models import ResNet18, wrn28_10_cifar10, wrn28_10_cifar100, wrn28_10  # use wrn28_10 for TinyImagenet200
from torchvision import transforms
from nbdt.utils import DATASET_TO_CLASSES, load_image_from_path, maybe_install_wordnet
from IPython.display import display

model = wrn28_10_cifar10()
model = SoftNBDT(
  pretrained=True,
  dataset='CIFAR10',
  arch='wrn28_10_cifar10',
  model=model,
  hierarchy='wordnet')

transforms = transforms.Compose([
  transforms.Resize(32),
  transforms.CenterCrop(32),
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

############### streamlit stuff ###############

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


if __name__ == '__main__':

    st.title('interpretable object recognition with NBDT')
    # Select a file
    if st.checkbox('Select a file in current directory'):
        folder_path = './data'
        # if st.checkbox('Change directory'):
        #     folder_path = st.text_input('Enter folder path', '.')
        filename = file_selector(folder_path=folder_path)
        st.write('You selected `%s`' % filename)
        st.image(filename, caption='3 marla plot')
        print(filename)

        im = load_image_from_path(filename)
        x = transforms(im)[None]

        # run inference
        outputs = model(
            x)  # to get intermediate decisions, use `model.forward_with_decisions(x)` and add `hierarchy='wordnet' to SoftNBDT
        st.write(outputs)
        st.write(outputs[0])
        st.write(outputs[0][0])

        confidence, predicted = outputs.max(1)
        st.write(outputs.max(1))

        cls = DATASET_TO_CLASSES['CIFAR10'][predicted[0]]

        st.write(f"prediction is {cls} with confidence {confidence.cpu().detach().numpy()[0]}")

        # now with hierarchy
        outputs = model.forward_with_decisions(x)
        st.write(outputs)
        st.write(outputs[1])
        st.write(outputs[1][0])



        pred = outputs[1]
        # confidence, predicted = pred.max(1)
        confidence, predicted = pred.max(1)
        # cls = DATASET_TO_CLASSES['CIFAR10'][predicted[0]]

        predicted = [DATASET_TO_CLASSES['CIFAR10'][p] for p in predicted]
        st.write(f'prediction = {predicted}')


    # TODO: Add code to open and process your image file
