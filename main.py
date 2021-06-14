import streamlit as st
import numpy as np
import os

from nbdt.model import SoftNBDT
from nbdt.models import ResNet18, wrn28_10_cifar10, wrn28_10_cifar100, wrn28_10  # use wrn28_10 for TinyImagenet200
from torchvision import transforms
from nbdt.utils import DATASET_TO_CLASSES, load_image_from_path, maybe_install_wordnet
from IPython.display import display
import base64

model = wrn28_10_cifar10()
model = SoftNBDT(
  pretrained=True,
  dataset='CIFAR10',
  arch='wrn28_10_cifar10',
  model=model,
  hierarchy='wordnet')


def set_bg_hack():
    # set bg name
    main_bg = "nbdt_bg.png"
    main_bg_ext = "png"

    # we can add a side bg if necessary
    # side_bg = "sample.jpg"
    # side_bg_ext = "jpg"

    st.markdown(
        f"""
         <style>
         .reportview-container {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})

         }}
         </style>
         """,
        unsafe_allow_html=True
    )


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
    set_bg_hack()
    st.title('interpretable object recognition with NBDT')
    st.write(f'NBDT stands for Neural Backed Decision Tree, and is a method to get more insight into the networks prediction. With minimal cost in accuracy the network can return a hierarchy representation of intermediate classifications.')
    st.write(f'read more in the paper: Wan et al., NBDT: Neural-Backed Decision Trees (2020)')
    # Select a file
    if st.checkbox('Select a file in current directory'):
        folder_path = './data'
        # if st.checkbox('Change directory'):
        #     folder_path = st.text_input('Enter folder path', '.')
        filename = file_selector(folder_path=folder_path)
        # st.write('You selected `%s`' % filename)
        st.image(filename, caption=f'{filename}')
        im = load_image_from_path(filename)
        x = transforms(im)[None]

        # run inference
        outputs = model(
            x)  # to get intermediate decisions, use `model.forward_with_decisions(x)` and add `hierarchy='wordnet' to SoftNBDT

        confidence, predicted = outputs.max(1)
        cls = DATASET_TO_CLASSES['CIFAR10'][predicted[0]]

        # now with hierarchy
        outputs, decisions = model.forward_with_decisions(
            x
        )  # use `model(x)` to obtain just logits
        confidence, predicted = outputs.max(1)
        cls = DATASET_TO_CLASSES["CIFAR10"][predicted[0]]
        st.header("prediction")
        st.write(f'{cls}')
        st.header("confidence")
        st.write(f'{confidence.cpu().detach().numpy()[0]}')

        classes = [item['name'] for item in decisions[0]]
        hierarchy = ' --> '.join(classes)
        st.header("hierarchy")
        st.write(f'{hierarchy}')


    # TODO: Add code to open and process your image file
