from comfy.latent_formats import LatentFormat

class LightdreamFormat(LatentFormat):
    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor
        self.latent_rgb_factors = [
                    #   R        G        B
                    [ 1.0,  0.0,  0.0],
                    [ 0.0,  1.0,  0.0],
                    [ 0.0,  0.0,  1.0]
                ]
        self.taesd_decoder_name = "taesd_decoder.pth"
