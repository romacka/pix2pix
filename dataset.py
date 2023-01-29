def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


class AnimeSketchDataset(Dataset):
    def __init__(self, img_paths):
        super().__init__()
        self.img_paths = img_paths
        self.transform_both = A.Compose([A.Resize(256, 256), A.HorizontalFlip()],
                                        additional_targets={"image0": "image"})
        self.transform_only_input = A.Compose(
            [A.ColorJitter(p=0.1), A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
             ToTensorV2(), ])
        self.transform_only_target = A.Compose(
            [A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ), ToTensorV2(), ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = np.asarray(Image.open(self.img_paths[idx]))

        sketchs = image[:, image.shape[1] // 2:, :]
        colored_imgs = image[:, :image.shape[1] // 2, :]

        aug = self.transform_both(image=sketchs, image0=colored_imgs)
        sketch, img = self.transform_only_input(image=aug["image"])['image'], \
                      self.transform_only_target(image=aug["image0"])['image']

        return sketch, img
