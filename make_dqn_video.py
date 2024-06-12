from xai import *
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import gc
import cv2
import argparse

def generate_video(
        dqn_model_path: str,
        device = "cpu",
        org = (10,4*207),
        fontscale = 4,
        thickness = 3,
        exploration_rate = 0.0,
        frame_skips = 4,
        number_of_frames = 3000,
        background_size = 50):

    dqn = DQN.load(dqn_model_path, device=device)

    with Recorder("dqn-agent.mp4", scale=1, fps=24) as recorder:
        with Window("Asteroids", 24, 1) as window:
            obs_background = dqn._autoencoder.encoder(torch.load("git-ignore/observations.pt")).output()
            state_background = torch.from_numpy(np.load("git-ignore/states.npy"))

            for step in dqn.rollout(exploration_rate, frame_skips=frame_skips).take(number_of_frames).monitor("Frame:", expected_length=number_of_frames):
                gc.collect()
                original = step.observation.numpy(False).repeat(4,0).repeat(4,1)
                transformed = step.observation.translated().rotated().numpy(False)
                reconstruction = dqn._autoencoder(transformed.astype(float)/255.0).output().numpy(force=True)
                transformed = transformed.repeat(4,0).repeat(4,1)
                reconstruction = (reconstruction*255).astype(np.uint8).repeat(4,0).repeat(4,1)
                q_values = dqn._policy(step.next_state).output().numpy(force=True)

                original = cv2.putText(
                    img=original, 
                    text=f"Original", 
                    org=org , 
                    fontFace=1,  
                    fontScale=fontscale, 
                    color=(255,255,255), 
                    thickness=thickness, 
                    lineType=cv2.LINE_AA
                    ) 
                
                transformed = cv2.putText(
                    img=transformed, 
                    text=f"Observation", 
                    org=org , 
                    fontFace=1,  
                    fontScale=fontscale, 
                    color=(255,255,255), 
                    thickness=thickness, 
                    lineType=cv2.LINE_AA
                    ) 
                
                reconstruction = cv2.putText(
                    img=reconstruction, 
                    text=f"Reconstruction", 
                    org=org , 
                    fontFace=1,  
                    fontScale=fontscale, 
                    color=(255,255,255), 
                    thickness=thickness, 
                    lineType=cv2.LINE_AA
                    ) 

                explanations = step.explain(
                    algorithm="permutation",
                    decoder_background=obs_background[torch.randperm(len(obs_background))[:background_size]],
                    q_background=state_background[torch.randperm(len(state_background))[:background_size]]
                )

                eap_shap_values = explanations.eap_explanation.shap_values

                eap_shap_values = np.append(eap_shap_values, eap_shap_values.sum(0, keepdims=True), axis=0)
                norm = np.max(np.abs(eap_shap_values))

                eap_images = np.zeros((len(eap_shap_values),210,160,3), dtype=np.uint8)
                black = np.zeros_like(eap_shap_values)
                red = np.where(eap_shap_values > 0, eap_shap_values/norm, black)
                blue = np.where(eap_shap_values < 0, -eap_shap_values/norm, black)
                eap_images[:,:,:,0] = (red*255).astype(np.uint8)
                eap_images[:,:,:,2] = (blue*255).astype(np.uint8)

                actions = (
                    "NOOP",
                    "UP",
                    "LEFT",
                    "RIGHT",
                    "FIRE"
                    )

                eap_images = eap_images.repeat(4,1).repeat(4,2)

                for i,eap_image in enumerate(eap_images[:-1]):
                    eap_images[i] = cv2.putText(
                        img=eap_images[i], 
                        text=f"Q-{actions[i]}: {q_values[i]:.2f}", 
                        org=org , 
                        fontFace=1,  
                        fontScale=fontscale, 
                        color=(255,255,255), 
                        thickness=thickness, 
                        lineType=cv2.LINE_AA
                        ) 
                    
                    
                eap_images[-1] = cv2.putText(
                    img=eap_images[-1], 
                    text=f"Q-sum", 
                    org=org , 
                    fontFace=1,  
                    fontScale=fontscale, 
                    color=(255,255,255), 
                    thickness=thickness, 
                    lineType=cv2.LINE_AA
                    ) 

                ecp_shap_values = explanations.ecp_explanation.shap_values

                ecp_shap_values = np.append(ecp_shap_values, ecp_shap_values.sum(0, keepdims=True), axis=0)
                norm = np.max(np.abs(ecp_shap_values))

                ecp_images = np.zeros((len(ecp_shap_values),210,160,3), dtype=np.uint8)

                red = np.where(ecp_shap_values > 0, ecp_shap_values/norm, black)
                blue = np.where(ecp_shap_values < 0, -ecp_shap_values/norm, black)
                ecp_images[:,:,:,0] = (red*255).astype(np.uint8)
                ecp_images[:,:,:,2] = (blue*255).astype(np.uint8)

                ecp_images = ecp_images.repeat(4,1).repeat(4,2)

                for i,ecp_image in enumerate(ecp_images[:-1]):
                    ecp_images[i] = cv2.putText(
                        img=ecp_images[i], 
                        text=f"Q-{actions[i]}: {q_values[i]:.2f}", 
                        org=org , 
                        fontFace=1,  
                        fontScale=fontscale, 
                        color=(255,255,255), 
                        thickness=thickness, 
                        lineType=cv2.LINE_AA
                        ) 
                    
                ecp_images[-1] = cv2.putText(
                    img=ecp_images[-1], 
                    text=f"Q-sum", 
                    org=org , 
                    fontFace=1,  
                    fontScale=fontscale, 
                    color=(255,255,255), 
                    thickness=thickness, 
                    lineType=cv2.LINE_AA
                    ) 


                l_shap_values = explanations.latent_explanation.flip().shap_values.sum((0,3))
                norm = np.max(np.abs(l_shap_values))

                l_image = np.zeros((210,160,3), dtype=np.uint8)
                black = np.zeros_like(l_shap_values)
                red = np.where(l_shap_values > 0, l_shap_values/norm, black)
                blue = np.where(l_shap_values < 0, -l_shap_values/norm, black)
                l_image[:,:,0] = (red*255).astype(np.uint8)
                l_image[:,:,2] = (blue*255).astype(np.uint8)

                l_image = l_image.repeat(4,0).repeat(4,1)

                l_image = cv2.putText(
                    img=l_image, 
                    text=f"latent-SHAP", 
                    org=org , 
                    fontFace=1,  
                    fontScale=fontscale, 
                    color=(255,255,255), 
                    thickness=thickness, 
                    lineType=cv2.LINE_AA
                    ) 

                black_image = np.zeros_like(original)

                image = np.hstack([
                    np.vstack([original,transformed,reconstruction]),
                    np.vstack([black_image,l_image,black_image]),
                    np.vstack(eap_images[:3]),
                    np.vstack(eap_images[3:]),
                    np.vstack(ecp_images[:3]),
                    np.vstack(ecp_images[3:]),
                    ])
                window(image)
                recorder(image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sample argument parser")
    parser.add_argument("dqn_path")
    args = parser.parse_args()
    generate_video(
        dqn_model_path=args.dqn_path
    )