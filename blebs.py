def pairwise_line(img, a, b):
    # a and b and 2 defects.
    # Finds a minimal line

    a = np.stack(np.where(a == 1), axis=0)
    b = np.stack(np.where(b == 1), axis=0)
    diff = a[:, :, None] - b[:, None, :]
    d_pairs = np.sqrt(np.sum(diff ** 2, axis=0))
    d = np.min(d_pairs)

    if d < 25:

        line = np.where(d_pairs == d)
        start = a[:, line[0]]
        start = np.flip(start.T[0])
        end = b[:, line[1]]
        end = np.flip(end.T[0])

        img_copy = np.ascontiguousarray(img, dtype=np.uint8)
        cv.line(img_copy, start, end, 0, 1)
        return img_copy

    else:
        return img


def bleb_out(img):
    segments, n_segments = label(img)
    valid = np.zeros(img.shape)
    for s in range(1, n_segments + 1):
        seg = segments == s

        hull = convex_hull_image(seg)
        defects = (1 - seg) * hull

        defects, n_defects_raw = label(defects)
        defect_mask = np.zeros(seg.shape)
        for i in range(1, n_defects_raw):
            defect_size = np.sum(defects == i)

            if defect_size >= 10:  # cutoff for defect registartion is 10 pixels...
                defect_mask = defect_mask + (defects == i)

        defects, n_defects_raw = label(defect_mask)

        for i in range(1, n_defects_raw + 1):
            for j in range(1, n_defects_raw + 1):
                if i == j:
                    continue

                seg = pairwise_line(seg, defects == i, defects == j)

            # cv.line(seg, start, end, 0, 1)

        valid = valid + seg

    return valid