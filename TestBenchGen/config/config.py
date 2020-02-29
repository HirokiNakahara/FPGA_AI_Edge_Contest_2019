former_lr_decay = 0.1

steplr_epoch = 10
steplr_factor = 0.5
#decay_lrs = {
#    60: 0.00001,
#    90: 0.000001
#}

anchors = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]

object_scale = 5
noobject_scale = 1
class_scale = 1
coord_scale = 1

saturation = 1.5
exposure = 1.5
hue = .1

jitter = 0.3

thresh = .6

momentum = 0.9
weight_decay = 0.0005


# multi-scale training:
# {k: epoch, v: scale range}
multi_scale = True

# number of steps to change input size
scale_step = 40

scale_range = (3, 4)

epoch_scale = {
    1:  (3, 4),
    15: (2, 5),
    30: (1, 6),
    60: (0, 7),
    75: (0, 9)
}

##alexnet
#input_sizes = [(327, 327),
#    (359, 359),
#    (391, 391),
#    (423, 423),
#    (455, 455),
#    (487, 487),
#    (519, 519),
#    (551, 551),
#    (583, 583)]

# darknet, vgg
input_sizes = [(320, 320),
               (352, 352),
               (384, 384),
               (416, 416),
               (448, 448),
               (480, 480),
               (512, 512),
               (544, 544),
               (576, 576)]

## alexnet
#input_size = (423, 423)
#test_input_size = (423, 423)

#darknet, vgg
input_size = (416, 416)
test_input_size = (416, 416)

gridH, gridW = 13, 13

debug = False

