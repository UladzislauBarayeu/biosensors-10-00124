classdef SwldaClassifier < handle
    %% Arno Libert 9-5-18
    
    properties
        targetLabels;
        description;
        C;
    end
    
    properties (Access = private)
        normalization_max;
        normalization_min;
    end
    
    methods
        function this = SwldaClassifier(labels)
            this.targetLabels = labels;
            this.C = [];
            this.description = 'SWLDA';
        end
        
        %% TRAIN
		% Trains the classifier with the given data
		% INPUT:	epochs (nb_electrodes x nb_samples x nb_epochs)
		%			labels (nb_epochs x 1)
		%				Vector indicating which labels have the corresponding epochs
		%           cues (nb_epochs x 1)
        %               Vector indicating which label is the target
		function train(this, epochs, labels, cues, ~)
%             features = this.extractFeatures(epochs);
            %% Vlad: 
            % if you load your features into features here and load in
            % which trials you want to train then this will work
			features = this.normalize_train_features(features);
			ntrain = size(features,1);
			xTrain = [features ones(ntrain,1)];
            
            % Construct the class labels
			is_target =  (labels == cues);
			yTrain = zeros(size(is_target,1),1);

			yTrain(is_target == 1) = 1; 
			yTrain(is_target == 0) = -1;
            %% THIS IS THE IMPORTANT BIT OF THE TRAINING
            props= {'MaxVar'      size(yTrain,1) 'INT'
                    'PEntry'      0.1         'DOUBLE'
                    'PRemoval'    0.15        'DOUBLE'
                   };

            maxVar = cell2mat(props(1,2));
            pEntry = cell2mat(props(2,2));
            pRemoval = cell2mat(props(3,2));

            if maxVar<1 | maxVar>size(yTrain,1),
              error(['limiting parameter of setpwise procedure MAXVAR must be between 1 and ' num2str(size(yTrain,1))]);
            end
            
            
            [b, se, pval, inmodel, stats]= ...
                        stepwisefit(xTrain, yTrain, 'penter',pEntry, 'premove',pRemoval, ...
                                    'maxiter',maxVar, 'display','off');
            this.C.w= zeros(size(b));
            this.C.w(inmodel)= b(inmodel);
            this.C.b = -this.C.w'*b;
        end
        
        function prediction = classify(this, epochs, labels)
			uniqueLabels = unique(labels);
			score = zeros(1, numel(uniqueLabels));
			for l = 1 : numel(uniqueLabels)
				% average per label
				avgdEpochs = mean(epochs(:,:,labels == l),3);

				% transform and normalize
				features = this.extractFeatures(avgdEpochs);
				features = this.normalize_test_features(features);
				ntest = size(features,1);
				Xtest = [features ones(ntest,1)];

				% Predict
                %% THIS IS THE IMPORTANT BIT OF THE CLASSIFIER
				transformed = Xtest * this.C.w;
	            % prediction  = (transformed > 0);
	            score(l) = transformed;
			end

			[~, predictionIdx] = max(score);
			prediction = uniqueLabels(predictionIdx);
		end
        
    end
    
    methods (Access = private)
        function features = extractFeatures(~, epochs)
			ep_length = size(epochs,2) * size(epochs,1);
			features = zeros(size(epochs,3), ep_length);
			for i = 1 : size(epochs, 3)
				features(i,:) = reshape(epochs(:,:,i)', 1, ep_length);
			end
        end
        
        function normalized = normalize_train_features(this, features)
            this.normalization_max = max(features);
			this.normalization_min = min(features);
			normalized  = (features - repmat(this.normalization_min,size(features,1),1)) ./ repmat(this.normalization_max-this.normalization_min,size(features,1),1);
        end
        
        function normalized = normalize_test_features(this, features)
            assert(~isempty(this.normalization_max));
            assert(~isempty(this.normalization_min));
			normalized  = (features - repmat(this.normalization_min,size(features,1),1)) ./ repmat(this.normalization_max-this.normalization_min,size(features,1),1);
        end
    end
    

end
