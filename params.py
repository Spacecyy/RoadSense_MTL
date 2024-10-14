
from torchvision import transforms



config= {
    'Campus_multi-domain_rtype':{
        'task_list':['noon','dusk','night'],
        'data_list':{'train':
                     {
                    'noon':
                     {
                    '5':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/noon/r_type/train_5',
                    '10':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/noon/r_type/train_10',
                    '20':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/noon/r_type/train_20',
                    '100':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/noon/r_type/train_100'
                    },
                    'dusk':
                    {
                    '5':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/dusk/r_type/train_5',
                    '10':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/dusk/r_type/train_10',
                    '20':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/dusk/r_type/train_20',
                    '100':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/dusk/r_type/train_100'
                    },
                    'night':
                    {
                    '5':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/night/r_type/train_5',
                    '10':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/night/r_type/train_10',
                    '20':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/night/r_type/train_20',
                    '100':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/night/r_type/train_100'
                    }
                    },

                    'test':
                    {
                    'noon':
                     {'5':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/noon/r_type/test_100',
                    '10':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/noon/r_type/test_100',
                    '20':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/noon/r_type/test_100',
                    '100':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/noon/r_type/test_100'},
                    'dusk':
                    {'5':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/dusk/r_type/test_100',
                    '10':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/dusk/r_type/test_100',
                    '20':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/dusk/r_type/test_100',
                    '100':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/dusk/r_type/test_100'},
                    'night':
                    {'5':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/night/r_type/test_100',
                    '10':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/night/r_type/test_100',
                    '20':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/night/r_type/test_100',
                    '100':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/night/r_type/test_100'}
                     },

                    'valid':
                    {
                    'noon':
                     {'5':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/noon/r_type/valid_100',
                    '10':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/noon/r_type/valid_100',
                    '20':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/noon/r_type/valid_100',
                    '100':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/noon/r_type/valid_100'},
                    'dusk':
                    {'5':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/dusk/r_type/valid_100',
                    '10':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/dusk/r_type/valid_100',
                    '20':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/dusk/r_type/valid_100',
                    '100':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/dusk/r_type/valid_100'},
                    'night':
                    {'5':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/night/r_type/valid_100',
                    '10':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/night/r_type/valid_100',
                    '20':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/night/r_type/valid_100',
                    '100':'/home/your_name/mtl/campus_crop_image/data_list/multi-domain/night/r_type/valid_100'}
                     }
                     },

        'train_transform':          
            transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(.4, .4, .4, .4),
                    transforms.ToTensor(),

                    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)), 
                    transforms.RandomErasing(p=0.5, scale=(0.22, 0.83), ratio=(0.3, 3.3)),


                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
        'test_transform':
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        'batch_size':32,
        'valid_batch_size':32,
        'test_batch_size':1,
        'num_classes':3,
    },


    'RSCD_multi-domain_rtype':{
        'task_list':['dry','wet','water'],
        'data_list':{'train':
                     {'dry':
                     {'5':'/home/your_name/RSCD-complete/data_list/multi-domain/dry/r_type/train_5',
                    '10':'/home/your_name/RSCD-complete/data_list/multi-domain/dry/r_type/train_10',
                    '20':'/home/your_name/RSCD-complete/data_list/multi-domain/dry/r_type/train_20',
                    '100':'/home/your_name/RSCD-complete/data_list/multi-domain/dry/r_type/train_100'},
                    'wet':
                    {'5':'/home/your_name/RSCD-complete/data_list/multi-domain/wet/r_type/train_5',
                    '10':'/home/your_name/RSCD-complete/data_list/multi-domain/wet/r_type/train_10',
                    '20':'/home/your_name/RSCD-complete/data_list/multi-domain/wet/r_type/train_20',
                    '100':'/home/your_name/RSCD-complete/data_list/multi-domain/wet/r_type/train_100'},
                    'water':
                    {'5':'/home/your_name/RSCD-complete/data_list/multi-domain/water/r_type/train_5',
                    '10':'/home/your_name/RSCD-complete/data_list/multi-domain/water/r_type/train_10',
                    '20':'/home/your_name/RSCD-complete/data_list/multi-domain/water/r_type/train_20',
                    '100':'/home/your_name/RSCD-complete/data_list/multi-domain/water/r_type/train_100'}
                    },

                    'valid':
                    {'dry':
                    {'5':'/home/your_name/RSCD-complete/data_list/multi-domain/dry/r_type/valid_5',
                    '10':'/home/your_name/RSCD-complete/data_list/multi-domain/dry/r_type/valid_10',
                    '20':'/home/your_name/RSCD-complete/data_list/multi-domain/dry/r_type/valid_20',
                    '100':'/home/your_name/RSCD-complete/data_list/multi-domain/dry/r_type/valid_100'},
                    'wet':
                    {'5':'/home/your_name/RSCD-complete/data_list/multi-domain/wet/r_type/valid_5',
                    '10':'/home/your_name/RSCD-complete/data_list/multi-domain/wet/r_type/valid_10',
                    '20':'/home/your_name/RSCD-complete/data_list/multi-domain/wet/r_type/valid_20',
                    '100':'/home/your_name/RSCD-complete/data_list/multi-domain/wet/r_type/valid_100'},
                    'water':
                    {'5':'/home/your_name/RSCD-complete/data_list/multi-domain/water/r_type/valid_5',
                    '10':'/home/your_name/RSCD-complete/data_list/multi-domain/water/r_type/valid_10',
                    '20':'/home/your_name/RSCD-complete/data_list/multi-domain/water/r_type/valid_20',
                    '100':'/home/your_name/RSCD-complete/data_list/multi-domain/water/r_type/valid_100'}},

                    'test':
                    {'dry':
                    {'5':'/home/your_name/RSCD-complete/data_list/multi-domain/dry/r_type/test_5',
                     '10':'/home/your_name/RSCD-complete/data_list/multi-domain/dry/r_type/test_10',
                     '20':'/home/your_name/RSCD-complete/data_list/multi-domain/dry/r_type/test_100',
                     '100':'/home/your_name/RSCD-complete/data_list/multi-domain/dry/r_type/test_100'},
                    'wet':
                    {'5':'/home/your_name/RSCD-complete/data_list/multi-domain/wet/r_type/test_5',
                     '10':'/home/your_name/RSCD-complete/data_list/multi-domain/wet/r_type/test_10',
                     '20':'/home/your_name/RSCD-complete/data_list/multi-domain/wet/r_type/test_100',
                     '100':'/home/your_name/RSCD-complete/data_list/multi-domain/wet/r_type/test_100'
                     },
                    'water':
                    {'5':'/home/your_name/RSCD-complete/data_list/multi-domain/water/r_type/test_5',
                     '10':'/home/your_name/RSCD-complete/data_list/multi-domain/water/r_type/test_10',
                     '20':'/home/your_name/RSCD-complete/data_list/multi-domain/water/r_type/test_20',
                     '100':'/home/your_name/RSCD-complete/data_list/multi-domain/water/r_type/test_100'}
                     }},

        'train_transform':          
            transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(.4, .4, .4, .4),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
        'test_transform':
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        'batch_size':32,
        'valid_batch_size':32,
        'test_batch_size':32,
        'num_classes':3,
        'label_catogories':['smooth', 'slight', 'severe']
    },

    
    }
