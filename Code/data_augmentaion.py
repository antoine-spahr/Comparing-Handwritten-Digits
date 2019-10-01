import torch

def ExtendData(x, t, as_list=True):
    #<*> DOCSTRING
    '''
        increase the number of image of a set by shiffting the original image by 1px in eight directions

        INPUT : x -> [torch.Tensor] n_sample x dim1 x dim2 : input set to be extended
                t -> [torch.Tensor] n_samplex1 : target asscociated with the input set to be extended

        OUTPUT : data -> [torch.Tensor] 9*n_sample x dim1 x dim2 : extended data by 9 folds
                 target -> [torch.Tensor] 9*n_sample x 1 : extended target


        _______      _______     _______     _______     _______     _______     _______     _______     _______
       |       |    | 0     |   |   0   |   |     0 |   |       |   |       |   |       |   |       |   |       |
       |   0   |    |       |   |       |   |       |   | 0     |   |     0 |   |       |   |       |   |       |
       |_______|    |_______|   |_______|   |_______|   |_______|   |_______|   |_0_____|   |___0___|   |_____0_|
       original      shift1      shift2      shift3      shift4      shift5      shift6      shift7      shift8
    '''
    # <*!>

    list_data = [torch.zeros(x.size()) for k in range(9)]
    list_target = [t for k in range(9)]

    # make the 9 possible shift of 1 pixel
    list_data[0][:,:-1,:-1] = x[:,1:,1:]
    list_data[1][:,:-1,:] = x[:,1:,:]
    list_data[2][:,:-1,1:] = x[:,1:,:-1]
    list_data[3][:,:,:-1] = x[:,:,1:]
    list_data[4][:,:,:] = x[:,:,:]
    list_data[5][:,:,1:] = x[:,:,:-1]
    list_data[6][:,1:,:-1] = x[:,:-1,1:]
    list_data[7][:,1:,:] = x[:,:-1,:]
    list_data[8][:,1:,1:] = x[:,:-1,:-1]

    if as_list:
        return list_data, list_target
    else:
        return torch.cat(list_data, dim=0), torch.cat(list_target, dim=0)

def ExtendPairData(x, t, c, mode='random_permutation'):
    #<*> DOCSTRING
    '''
        increase the number of image of a paired image set (n_sample x 2 x dim1 x dim2) by shiffting the original image by 1px in eight directions

        INPUT : x -> [torch.Tensor] n_sample x 2 x dim1 x dim2 : input set to be extended
                t -> [torch.Tensor] n_samplex1 : target (link between the two imgs) asscociated with the input set to be extended
                c -> [torch.Tensor] n_samplex2 : class asscociated with each of the two images of the input
                mode -> [string] define how to extend the pair. Three option are possibles
                                    1) 'no_permuattion' : the image shifted by ExtendData are paired together
                                    2) 'random_permutation' : the image shifted by ExtendData are paired randomly with other shift
                                    3) 'all_permutation' : the image shifted by ExtendData are combined to form all the possible pair of shift -> increases the number of data by 9-fold

        OUTPUT : data -> [torch.Tensor] 9*n_sample x dim1 x dim2 : extended data by 9 folds
                 target -> [torch.Tensor] 9*n_sample x 1 : extended target
                 clss -> [torch.Tensor] 9*n_sample x 2 : extended class
    '''
    # <*!>

    if mode == 'no_permutation':
        img1_data_extended, img1_class_extended = ExtendData(x[:,0,:,:], c[:,0], as_list=False)
        img2_data_extended, img2_class_extended = ExtendData(x[:,1,:,:], c[:,1], as_list=False)

        data = torch.stack([img1_data_extended, img2_data_extended], dim=1)
        clss = torch.stack([img1_class_extended, img2_class_extended], dim=1)
        target = torch.cat([t for k in range(9)], dim=0)

        return data, target, clss

    elif mode == 'random_permutation':
        img1_data_extended, img1_class_extended = ExtendData(x[:,0,:,:],c[:,0],as_list=True)
        img2_data_extended, img2_class_extended = ExtendData(x[:,1,:,:],c[:,1],as_list=True)

        tmp1 = torch.stack(img1_data_extended, dim=3)
        tmp2 = torch.stack(img2_data_extended, dim=3)

        data_all = torch.stack([tmp1,tmp2], dim=1)

        for i in range(data_all.size(0)):
            data_all[i,0,:,:,:] = data_all[i,0,:,:,torch.randperm(9)]
            data_all[i,1,:,:,:] = data_all[i,1,:,:,torch.randperm(9)]

        data_all_list = [data_all[:,:,:,:,i].squeeze() for i in range(data_all.size(4))]
        data = torch.cat(data_all_list, dim=0)
        clss = torch.stack([torch.cat(img1_class_extended, dim=0), torch.cat(img2_class_extended, dim=0)], dim=1)
        target = torch.cat([t for k in range(len(img1_data_extended))], dim=0)

        return data, target, clss

    elif mode == 'all_permutation':
        img1_data_extended, img1_class_extended = ExtendData(x[:,0,:,:],c[:,0],as_list=True)
        img2_data_extended, img2_class_extended = ExtendData(x[:,1,:,:],c[:,1],as_list=True)

        data_tmp1 = []
        data_tmp2 = []
        clss_tmp1 = []
        clss_tmp2 = []

        for i in range(len(img1_data_extended)):
            for j in range(len(img2_data_extended)):
                data_tmp1.append(img1_data_extended[i])
                data_tmp2.append(img2_data_extended[j])
                clss_tmp1.append(img1_class_extended[i])
                clss_tmp2.append(img2_class_extended[j])

        data_tmp1 = torch.cat(data_tmp1, dim=0)
        data_tmp2 = torch.cat(data_tmp2, dim=0)
        clss_tmp1 = torch.cat(clss_tmp1, dim=0)
        clss_tmp2 = torch.cat(clss_tmp2, dim=0)

        data = torch.stack([data_tmp1,data_tmp2], dim=1)
        clss = torch.stack([clss_tmp1, clss_tmp2], dim=1)
        target = torch.cat([t for k in range(len(img1_data_extended)*len(img2_data_extended))], dim=0)

        return data, target, clss

    else:
        raise ValueError("Wrong mode choosen! must be 'no_permutation' or 'random_permutation' or 'all_permutation'")

def FlipImages(x, t, c):
    #<*> DOCSTRING
    '''
        Double the data by inversing the order of the two images, class and target and merging those new data with the old ones

        INPUT : x -> [torch.Tensor] n_sample x 2 x dim1 x dim2 : input set to be extended
                t -> [torch.Tensor] n_samplex1 : target (link between the two imgs) asscociated with the input set to be extended
                c -> [torch.Tensor] n_samplex2 : class asscociated with each of the two images of the input

        OUTPUT : data -> [torch.Tensor] 9*n_sample x dim1 x dim2 : extended data by 9 folds
                 target -> [torch.Tensor] 9*n_sample x 1 : extended target
                 clss -> [torch.Tensor] 9*n_sample x 2 : extended class
    '''
    #<*!>
    # flip data class and target
    data = torch.cat((x, x.flip(1)), dim=0)
    flip_c = c.flip(1)
    clss = torch.cat((c, flip_c), dim=0)
    target = torch.cat((t, (flip_c[:,0] <= flip_c[:,1]).long()),dim=0)

    return data, target, clss
