#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }


echo -e "\nYou need to register at https://bedlam2.is.tue.mpg.de/"
read -p "Username (BEDLAM2):" username
read -p "Password (BEDLAM2):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p data/pretrained-models
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam2&sfile=checkpoints/camerahmr/bedlam_v1_v2.ckpt' -O './data/pretrained-models/bedlam_v1_v2.ckpt' --no-check-certificate --continue

echo -e "\nYou need to register at https://camerahmr.is.tue.mpg.de/"
read -p "Username (CameraHMR):" username
read -p "Password (CameraHMR):" password
username=$(urle $username)
password=$(urle $password)

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=cam_model_cleaned.ckpt' -O './data/pretrained-models/cam_model_cleaned.ckpt' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=model_final_f05665.pkl' -O './data/pretrained-models/model_final_f05665.pkl' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=smpl_mean_params.npz' -O './data/smpl_mean_params.npz' --no-check-certificate --continue

# # SMPL-X model
echo -e "\nYou need to register at https://smpl-x.is.tue.mpg.de"
read -p "Username (SMPL-X):" username
read -p "Password (SMPL-X):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p data/models/smplx_neutral_head
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_lockedhead_20230207.zip' -O './data/models/smplx_lockedhead_20230207.zip' --no-check-certificate --continue
unzip data/models/smplx_lockedhead_20230207.zip -d data/models/smplx_neutral_head
