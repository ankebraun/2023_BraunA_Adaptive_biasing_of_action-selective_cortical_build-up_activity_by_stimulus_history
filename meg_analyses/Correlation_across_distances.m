% Adapted from Alessandro Toso.
% This code correlates the lcmv spatial filters of pairs of sources at
% different distances ( from 0 to 5 cm), to estimate how much  filters
% correlation scales with space


clear all
close all
subjects={'2'}%, '3', '5', '7', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '37', '38', '39', '40', '41', '42'};
hemi=1; % 1 left 2 right
sessions = 1:3;
for s=1:length(subjects)
    for sess = sessions    
        L=readmatrix(['source_vx_id_sj_',cell2mat(subjects(s)),'_sess_',sprintf('%d',sess), '_L.csv']);
        R=readmatrix(['source_vx_id_sj_',cell2mat(subjects(s)),'_sess_',sprintf('%d',sess), '_R.csv']);
        L=L(2:end,2);
        R=R(2:end,2);
        if hemi==1
            id_filter=L;
        else
            id_filter=R;               
        end
        if hemi==1
            readmatrix(['source_vx_dists_sj_',cell2mat(subjects(s)),'_sess_',sprintf('%d',sess), '_L.csv']);
            dist_vx_source=ans(2:end,2:end);
        else
            readmatrix(['source_vx_dists_sj_',cell2mat(subjects(s)),'_sess_',sprintf('%d',sess), '_R.csv']);
            dist_vx_source=ans(2:end,2:end);    
        end
        %caluculate dist between sources
        vx_used=dist_vx_source(id_filter+1,:);
        x=vx_used(:,1);y=vx_used(:,2);z=vx_used(:,3);
        All_dist=[];
        for i=1:size(vx_used,1)
            for ii=1:size(vx_used,1)
                All_dist(i,ii)=sqrt((x(i)-x(ii))^2+(y(i)-y(ii))^2+(z(i)-z(ii))^2);
            end
        end

        % filters for all sources
        Corr=readmatrix(['All_source_filter_weights_sj_',cell2mat(subjects(s)),'_sess_',sprintf('%d',sess), '.csv']);
        Corr=Corr(2:end,2:end);

        %  sources per hemi
        n_vx_left=size(not(isnan(L)),1);
        n_vx_right=size(not(isnan(R)),1);

        All_areas=[];
        if hemi==1
            All_areas=Corr(1:n_vx_left,:);
        else
            All_areas=Corr((n_vx_left+1):end,:);  
        end

        Correlations=[];
        Correlations=corrcoef(All_areas','Rows','complete');

        All_source=[];
        for i=1:size(Correlations,1)-1
            dist=[0:0.5:5];
        for ii=1:11
            id=find(round(All_dist(i,:)*100,1)==dist(ii));
            All_vx_side(ii,:)=mean((Correlations(i,id)));
        end 
            All_source(:,i)=All_vx_side;
        end

        All_sess(:,sess)=nanmean(All_source,2);
    end

    
    writematrix(nanmean(All_sess, 2), sprintf('Mean_All_sess_dist_subj_%s_hemi1.csv', subjects{s}));
    All_sub(:,s)=nanmean(All_sess,2);
        
end



 

