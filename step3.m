% excute the job in dqshtc

DIR= '/scratch/tli3/PSC_biobank/';
datadir=fullfile(DIR,'data');
moveto='/BIGS2DATA/Dataset/Non_MDAdata/BioBank/Imaging/ProcessedData/PSC_Tengfei_20180419/step3/';
subID=dir(datadir);
subID={subID.name};
subID=subID(3:end);
L=length(subID);

scriptdir =fullfile(DIR,'Scilpy/');
scriptdir1 =fullfile(scriptdir,'PSC_Pipeline/UK_Biobank/');

codedir = fullfile(DIR,'code23');

if exist(codedir)
   delete(fullfile(codedir,'/*'));
end
mkdir(codedir)

fid0 = fopen(fullfile(codedir,'All_step3.sh'),'w');
fprintf(fid0,'#!/bin/tcsh\n');
for i=1:L
    sub_id = subID{i};
	temp=fullfile(datadir,sub_id,'dwi.nii.gz')
	temp1=fullfile(datadir,sub_id,'data.nii.gz')
	temp2=fullfile(datadir,sub_id,'nodif.nii.gz')
	temp3=sprintf('%s/%s/pnc%s',datadir,sub_id,sub_id);
	if (~exist(temp)) & (~exist(temp1)) & (~exist(temp2))
	continue;
	end
	if ~exist(temp3)
	continue;
	end
	temp=fullfile(datadir,sub_id);
	temp0=dir(temp);
	temp0={temp0.name}';
	temp0=temp0(3:end);
	l00=length(temp0);
	temp=fullfile(datadir,sub_id,'diffusion');
	temp1=dir(temp);
	temp1={temp1.name}';
	temp1=temp1(3:end);
	l01=length(temp1);
	temp=fullfile(datadir,sub_id,'registration');
	temp1=dir(temp);
	temp1={temp1.name}';
	temp1=temp1(3:end);
	l02=length(temp1);
	temp=fullfile(datadir,sub_id,'structural');
	temp2=dir(temp);
	temp2={temp2.name}';
	temp2=temp2(3:end);
	l03=length(temp2);
	if (l01>36)&(l02==3)&(l03==19) 
	system(sprintf('mv %s/%s %s',datadir,subID{i},moveto));
	system(sprintf('rm %s/%s/diffusion/data.nii.gz',moveto,subID{i}))
	system(sprintf('rm %s/%s/diffusion/data_1x1x1.nii.gz',moveto,subID{i}))
	system(sprintf('rm %s/%s/diffusion/dwi_rnlm.nii.gz',moveto,subID{i}))
	continue;
	end
	
    fid = fopen(sprintf('%s/jobsubmission_stg3_%s.pbs',codedir,subID{i}),'w');
    %fprintf(fid,'#PBS -l nodes=1:rhel7:ppn=1 -l walltime=23:59:59,mem=8gb\n');
    fprintf(fid,'#PBS -l nodes=1:rhel7:ppn=4 -l walltime=23:59:59,mem=32gb\n');
    fprintf(fid,'#PBS -N Zhengwu_%s\n',subID{i});
    fprintf(fid,'#PBS -o Zhengwu_%s.out\n',subID{i});
    fprintf(fid,'#PBS -j oe\n');
	fprintf(fid, '# sub job for subject %s \n', sub_id);
    fprintf(fid,'module load gsl/1.16\n');
    fprintf(fid,'module load mrtrix\n');
    fprintf(fid,'module load FSL\n');
    fprintf(fid,'module load ANTs/2.1.0\n');
	fprintf(fid,'module load freesurfer\n');
	fprintf(fid,'setenv PATH /workspace/tli3/software/Anaconda2_for_scilpy2017/Anaconda2/bin:$PATH \n');
	fprintf(fid,'setenv PATH %s/scripts/:${PATH} \n',scriptdir);
    fprintf(fid,'setenv PYTHONPATH %s/ \n',scriptdir);
	fprintf(fid,'cd %s/%s \n',datadir,sub_id);
	fprintf(fid,'cp pnc%s/mri/aparc.a2009s+aseg.mgz ./ \n',sub_id);
    fprintf(fid,'cp pnc%s/mri/wmparc.mgz ./ \n',sub_id);
    fprintf(fid,'cp pnc%s/mri/rawavg.mgz ./ \n',sub_id);
	fprintf(fid,'cp pnc%s/mri/brainmask.mgz ./ \n',sub_id);
    %remove output1 and prepare for re-write
    fprintf(fid,'chmod 775 %s/TRACTOGRAPHY_Step1.sh \n',scriptdir1);
	fprintf(fid,'sh %s/TRACTOGRAPHY_Step1.sh \n',scriptdir1);
    fprintf(fid, '# sub job for subject %s is done \n', sub_id);
    fprintf(fid, '# -------------------------   #');
    fclose(fid);	
	fprintf(fid0,'qsub jobsubmission_stg3_%s.pbs\n',subID{i});
end

fclose(fid0);
