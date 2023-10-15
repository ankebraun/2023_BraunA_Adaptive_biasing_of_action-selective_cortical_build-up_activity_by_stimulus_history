% Adapted from Alessandro Toso.
% This code correlates the lcmv spatial filters of different sources between ROIs
% it takes a random source per ROI and correlate them, and iterates the
% process for *bt_n* times - for each subject. Then plot the average corr
% across subjects
clear all
close all
%subjects={'2', '3', '5', '7', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '37', '38', '39', '40', '41', '42'};
subjects={'2'}
bt_n=30;
areas = {'vfcPrimary', 'vfcEarly', 'vfcV3ab', 'vfcIPS01', 'vfcIPS23', 'JWG_aIPS', 'vfcLO', 'vfcTO', 'vfcVO', 'vfcPHC', 'JWG_IPS_PCeS', 'JWG_M1'}

sessions = 1:3;

hemi=1; % select hemifield (1 left, 2 right)

All_sub = zeros(length(areas), length(areas),length(subjects));
for s=1:length(subjects)
    if isfile(sprintf('Mean_All_sess_corr_subj_%s_hemi1.csv', subjects{s}))
        All_sess_corr_mean = load(sprintf('Mean_All_sess_corr_subj_%s_hemi1.csv', subjects{s}));
        All_sub(:,:,s)=All_sess_corr_mean; 
    else
        All_sess_corr = zeros(length(areas), length(areas),length(sessions));

        for sess = sessions
            All_bt_corr = zeros(length(areas), length(areas), bt_n);
            L=readmatrix(['source_vx_id_sj_',cell2mat(subjects(s)),'_sess_',sprintf('%d',sess),'_L.csv']);
            R=readmatrix(['source_vx_id_sj_',cell2mat(subjects(s)),'_sess_',sprintf('%d',sess),'_R.csv']);
            L=L(2:end,2);
            R=R(2:end,2);
            if hemi==1
                id_filter=L;
            else
                id_filter=R;               
            end
            for bt=1:bt_n
                for a=1:length(areas) 
                    if hemi==1
                        readmatrix(['id_VX__sj_',cell2mat(subjects(s)),'_sess_',sprintf('%d',sess),'_area_',cell2mat(areas(a)),'_l.csv']);
                    else
                        readmatrix(['id_VX__sj_',cell2mat(subjects(s)),'_sess_',sprintf('%d',sess),'_area_',cell2mat(areas(a)),'_r.csv']);       
                    end
                    id_area=ans(2,2:end);
                    % select which sources belong to the ROI
                    source_area=[];
                    for i_1=1:size(id_area,2)
                        source_id=find(id_filter==id_area(i_1));
                        source_area(i_1)=(source_id);
                    end
                    % filters for all sources
                    Corr=readmatrix(['All_source_filter_weights_sj_',cell2mat(subjects(s)),'_sess_',sprintf('%d',sess),'.csv']);
                    % take relevat filters
                    Corr=Corr(source_area+1,2:end);
                    % take a random vx from  relvant filters
                    rand_id=randi(size(Corr,1),1);
                        if exist('All_areas', 'var') == 1
                            if size(Corr, 2) < size(All_areas, 2)
                                Corr(rand_id,end+1:size(All_areas, 2)) = missing;
                            elseif size(Corr, 2) > size(All_areas, 2)
                                All_areas(rand_id,end+1:size(Corr, 2)) = missing;                            
                            end
                        end
                    All_areas(a,:)=Corr(rand_id,:);
                end
                % correlation between filters from areas
                Correlations=[];
                Correlations=corrcoef(All_areas','Rows','complete');
                All_bt_corr(:,:,bt)=Correlations;
            end
            All_sess_corr(:,:,sess)=nanmean(All_bt_corr,3);
        end
        writematrix(nanmean(All_sess_corr, 3), sprintf('Mean_All_sess_corr_subj_%s_hemi1.csv', subjects{s}));
        All_sub(:,:,s)=nanmean(All_sess_corr,3);   
    end
end