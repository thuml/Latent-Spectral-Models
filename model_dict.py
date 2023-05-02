from models import LSM_2D, LSM_3D, LSM_Irregular_Geo, FNO_2D, FNO_3D, FNO_Irregular_Geo

def get_model(args):
    model_dict = {
        'FNO_2D': FNO_2D,
        'FNO_3D': FNO_3D,
        'FNO_Irregular_Geo': FNO_Irregular_Geo,
        'LSM_2D': LSM_2D,
        'LSM_3D': LSM_3D,
        'LSM_Irregular_Geo': LSM_Irregular_Geo,
    }
    if args.model == 'LSM_Irregular_Geo' or args.model == 'FNO_Irregular_Geo':
        return model_dict[args.model].Model(args).cuda(), model_dict[args.model].IPHI().cuda()
    else:
        return model_dict[args.model].Model(args).cuda()
