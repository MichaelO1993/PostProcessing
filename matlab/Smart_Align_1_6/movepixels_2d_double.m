function Iout=movepixels_2d_double(Iin,Tx,Ty,mode)
% This function movepixels, will translate the pixels of an image
%  according to x and y translation images (bilinear interpolated). 
% 
%  Iout = movepixels_2d_double(I,Tx,Ty,mode);
%
% Inputs;
%   Tx, Ty: The transformation images, describing the
%             (backwards) translation of every pixel in x and y direction.
%   mode: If 0: linear interpolation and outside pixels set to nearest pixel
%            1: linear interpolation and outside pixels set to zero
%            (cubic interpolation only supported by compiled mex file)
%            2: cubic interpolation and outsite pixels set to nearest pixel
%            3: cubic interpolation and outside pixels set to zero
%
% Outputs,
%   Iout : The transformed image
%
% Function is written by D.Kroon University of Twente (February 2009)

%% Decision if the Iin is HAADF or Index matrix
translation_saving = 0;  

global image_stack_translation;


if all(Iin(:) == 1)
   translation_saving = 1;
end



global i_count;
global indexing;


%% If HAADF the pixels are moved, if Index-Matrix the translation vectors are saved
if ~translation_saving
    
    % Make all x,y indices
    [x,y]=ndgrid(0:size(Iin,1)-1,0:size(Iin,2)-1);

    % Calculate the Transformed coordinates
    Tlocalx = x+Tx;
    Tlocaly = y+Ty;

    % All the neighborh pixels involved in linear interpolation.
    xBas0=floor(Tlocalx); 
    yBas0=floor(Tlocaly);
    xBas1=xBas0+1;          
    yBas1=yBas0+1;

    % Linear interpolation constants (percentages)
    xCom=Tlocalx-xBas0; 
    yCom=Tlocaly-yBas0;
    perc0=(1-xCom).*(1-yCom);
    perc1=(1-xCom).*yCom;
    perc2=xCom.*(1-yCom);
    perc3=xCom.*yCom;

    % limit indexes to boundaries
    check_xBas0=(xBas0<0)|(xBas0>(size(Iin,1)-1));
    check_yBas0=(yBas0<0)|(yBas0>(size(Iin,2)-1));
    xBas0(check_xBas0)=0; 
    yBas0(check_yBas0)=0; 
    check_xBas1=(xBas1<0)|(xBas1>(size(Iin,1)-1));
    check_yBas1=(yBas1<0)|(yBas1>(size(Iin,2)-1));
    xBas1(check_xBas1)=0; 
    yBas1(check_yBas1)=0; 


    Iout=zeros(size(Iin));
    for i=1:size(Iin,3)    
        Iin_one=Iin(:,:,i);
        % Get the intensities
        intensity_xyz0=Iin_one(1+xBas0+yBas0*size(Iin,1));
        intensity_xyz1=Iin_one(1+xBas0+yBas1*size(Iin,1)); 
        intensity_xyz2=Iin_one(1+xBas1+yBas0*size(Iin,1));
        intensity_xyz3=Iin_one(1+xBas1+yBas1*size(Iin,1));
        % Make pixels before outside Ibuffer mode
        if(mode==1||mode==3)
            intensity_xyz0(check_xBas0|check_yBas0)=0;
            intensity_xyz1(check_xBas0|check_yBas1)=0;
            intensity_xyz2(check_xBas1|check_yBas0)=0;
            intensity_xyz3(check_xBas1|check_yBas1)=0;
        end
        Iout_one=intensity_xyz0.*perc0+intensity_xyz1.*perc1+intensity_xyz2.*perc2+intensity_xyz3.*perc3;
        Iout(:,:,i)=reshape(Iout_one, [size(Iin,1) size(Iin,2)]);
    end

else
    
    indexing = indexing + 1;
    i_count{[indexing]} = indexing;

    
    translation_x = image_stack_translation{[2*indexing-1]};
    translation_y = image_stack_translation{[2*indexing]};
    translation_size = size(translation_x,3);
    if isempty(translation_x) 
        translation_x = Tx;
        translation_y = Ty;
    else
        translation_x(:,:,translation_size+1) = Tx;
        translation_y(:,:,translation_size+1) = Ty;
    end
    image_stack_translation{[2*indexing-1]} = translation_x;
    image_stack_translation{[2*indexing]} = translation_y;
    
    Iout = Iin;
end



