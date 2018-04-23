%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Process MRI data using dqshtc Group Analysis on BIOS Sever   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% March 19, 2012 @ by LLK
% Jan    8, 2015 @ by CH
% May   17, 2016 @ by TFL

FSDir = '/rsrch2/biostatistics/tli3/PNC_Zhengwu_freesurfer';                % directory containing the data folder and other files.  you need to change the directory accordingly
datadir=fullfile(FSDir,'data');
codedir=fullfile(FSDir,'code');
mkdir(codedir);
delete(fullfile(codedir,'*'));

subNames = dir(datadir);        % 'FS_TestData': name of directory containing the imaging data files.  you need to change the directory accordingly
subNames = {subNames.name}';
subNames = subNames(3:end); % first two are sup-directory and current one

nn = size(subNames,1);

fid0 = fopen(sprintf('%s/FS_batAll.sh',codedir),'w');
fprintf(fid0,'#!/bin/bash\n');


for ii=1:nn
    sub_id = subNames{ii};
	temp=sprintf('%s/%s/pnc%s/mri/aparc.a2009s+aseg.mgz',datadir,sub_id,sub_id);
	temp0=sprintf('%s/%s/pnc%s/mri/wmparc.mgz',datadir,sub_id,sub_id);
	if exist(temp) & exist(temp0)
	continue;
	end
	% temp=fullfile(datadir,sub_id,'dwi.nii.gz')
	% temp1=fullfile(datadir,sub_id,'data.nii.gz')
	% temp2=fullfile(datadir,sub_id,'nodif.nii.gz')
	% if (~exist(temp)) & (~exist(temp1)) & (~exist(temp2))
	% continue;
	% end
    batNames = sprintf('%s/FS_bat%i.pbs',codedir,ii);
    fid = fopen(batNames,'w'); 
	fprintf(fid,'#PBS -N freesurfer\n');
    fprintf(fid,'#PBS -l nodes=1:ppn=8,walltime=23:59:59,mem=40gb\n');
    fprintf(fid,'#PBS -o %s\n',codedir);
    fprintf(fid,'#PBS -e %s\n',codedir);
    fprintf(fid,'#PBS -d %s\n',codedir);
    fprintf(fid,'#PBS -V\n');
    fprintf(fid,'#PBS -m e\n');
    fprintf(fid,'#PBS -M YourEmailAddressHerer@server\n');
    fprintf(fid,'#This line is required.\n');
    fprintf(fid,'#export MCR_CACHE_ROOT=$TMPDIR\n');
    fprintf(fid,'\n');
    fprintf(fid,'module load freesurfer \n');
    fprintf(fid,'#!/bin/bash\n');
    fprintf(fid,'export FREESURFER_HOME=/risapps/rhel6/freesurfer/5.3.0/freesurfer\n');
    fprintf(fid,'source $FREESURFER_HOME/SetUpFreeSurfer.sh\n');
    fprintf(fid,'export SUBJECTS_DIR=%s/%s\n', datadir,subNames{ii}); 
	fprintf(fid,'cd %s/%s \n',datadir,subNames{ii});
	fprintf(fid,'rm -r -f pnc%s \n',subNames{ii});
	fprintf(fid,'recon-all -openmp 8 -subjid pnc%s -i T1_dti_final.nii.gz -all\n',subNames{ii});
	fclose(fid);
    fprintf(fid0,'msub FS_bat%i.pbs\n',ii);
end

fclose(fid0);

clear all;