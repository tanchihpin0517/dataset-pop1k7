import os 
import glob

def traverse_dir(
        root_dir,
        extension=('mid', 'MID', 'midi', 'pkl', 'npy'),
        amount=None,
        str_=None,
        is_pure=False,
        verbose=False,
        is_sort=False,
        is_ext=True):
    if verbose:
        print('[*] Scanning...')
    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                if (amount is not None) and (cnt == amount):
                    break
                if str_ is not None:
                    if str_ not in file:
                        continue
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                if verbose:
                    print(pure_path)
                file_list.append(pure_path)
                cnt += 1
    if verbose:
        print('Total: %d files' % len(file_list))
        print('Done!!!')
    if is_sort:
        file_list.sort()
    return file_list


if __name__ == '__main__':
    root_dir = "/Users/wayne391/Documents/Projects/mt-new/flickinger/dataset/representations/uncond/cp/ailab17k_from-scratch_cp/events"
    filelist = traverse_dir(root_dir)
    filelist.sort()

    for idx, f in enumerate(filelist):
        print(f'{idx}-------------')
        print(f)

        bn = os.path.basename(f)
        path_dir = f[:-len(bn)]
        ext = bn.split('.')[-1]

        outname = os.path.join(path_dir, f'{idx}.{ext}')
        print(outname)
        os.rename(f, outname)



