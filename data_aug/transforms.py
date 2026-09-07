import albumentations as A

def get_ssl_transform(config):
    """
    Menghasilkan fungsi transformasi fisika untuk pra-pelatihan SSL.
    DILARANG KERAS menggunakan Flip (H/V) atau Rotasi untuk menjaga kiralitas filamen.
    """
    ssl_conf = config.data_augmentation.ssl
    return A.Compose([
        A.RandomResizedCrop(
            size=(ssl_conf.crop_size, ssl_conf.crop_size), 
            scale=list(ssl_conf.crop_scale), 
            p=1.0
        ),
        A.ColorJitter(
            brightness=ssl_conf.color_jitter_brightness, 
            contrast=ssl_conf.color_jitter_contrast, 
            saturation=0, hue=0, 
            p=ssl_conf.jitter_prob
        ),
        A.GaussianBlur(
            blur_limit=(ssl_conf.blur_limit_min, ssl_conf.blur_limit_max), 
            p=ssl_conf.blur_prob
        )
    ])

def get_supervised_transform(config):
    """
    Menghasilkan fungsi transformasi deterministik untuk Supervised Fine-Tuning.
    """
    sup_conf = config.data_augmentation.supervised
    return A.Compose([
        A.Resize(size=(sup_conf.resize_dim, sup_conf.resize_dim))
    ])
