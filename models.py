from transvfs import build_transvfs 


def build_model(args): 
    model, criterion = None, None 
    if args.net_name == 'TransVFS': 
        model, criterion = build_transvfs(args) 
    else: 
        raise NotImplementedError("{} has not been implemented".format(args.net_name)) 
    return model, criterion 
