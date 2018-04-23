
####################cset scilpy's pathc###########
#### change to your own path

# cd to the data folder
cd connectome

###### extract the streamline connectivity matrix for both cortical and subcortical 

#desikan atlas
# extract connectivity matrices, get the dilation of images for Desikan
extraction_sccm_withfeatures_cortical.py ../streamlines/full_interface_prob_pft_invcoord.trk ../diffusion/fa.nii.gz ../diffusion/md.nii.gz ../structural/wmparc.nii.gz $SCIP_PATH/scilpy/connectome/Desikan_ROI.txt $SCIP_PATH/scilpy/connectome/FreeSurferColorLUT.txt UKBB 20 240 1 2 4 desikan
extraction_sccm_withfeatures_subcortical.py ../streamlines/full_interface_prob_pft_invcoord.trk ../diffusion/fa.nii.gz ../diffusion/md.nii.gz ../structural/wmparc.nii.gz UKBB_desikan_dilated_labels.nii.gz  $SCIP_PATH/scilpy/connectome/Subcortical_ROI.txt $SCIP_PATH/scilpy/connectome/FreeSurferColorLUT.txt UKBB 20 240 4 0 desikan


# Destreoux
extraction_sccm_withfeatures_cortical.py ../streamlines/full_interface_prob_pft_invcoord.trk ../diffusion/fa.nii.gz ../diffusion/md.nii.gz ../structural/aparc.a2009s+aseg.nii.gz $SCIP_PATH/scilpy/connectome/Destrieux_ROI.txt $SCIP_PATH/scilpy/connectome/FreeSurferColorLUT.txt UKBB 20 240 1 2 4 destrieux
extraction_sccm_withfeatures_subcortical.py ../streamlines/full_interface_prob_pft_invcoord.trk ../diffusion/fa.nii.gz ../diffusion/md.nii.gz ../structural/aparc.a2009s+aseg.nii.gz UKBB_destrieux_dilated_labels.nii.gz  $SCIP_PATH/scilpy/connectome/Subcortical_ROI.txt $SCIP_PATH/scilpy/connectome/FreeSurferColorLUT.txt UKBB 20 240 4 0 destrieux

