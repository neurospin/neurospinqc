#! /usr/bin/env python
##########################################################################
# Nsap - Neurospin - Berkeley - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import numpy


def mq(rigid, shape, spacing, origin, metric="max"):
    """ Movement quantity (MQ) - Description of the amount of motion.

    Parameters
    ----------
    rigid: array (4, 4)
        a rigid transformation.
    shape, sapcing, origin: array (3,)
        the image shape, spacing and origin.
    metric: str (optional, default 'max')
        the output metric type: 'max', 'mean' or 'stat' (mean and std).

    Returns
    -------
    mq: tuple
        the required global displacement metric.
    """
    # Compute the local displacement
    displacement = deformation_field(rigid, shape, spacing, origin)

    # Compute the norm of each displacement
    norm_displacement = numpy.sqrt(
        numpy.sum(displacement[..., i]**2 for i in range(3)))

    # Return the requested metric
    if metric == "max":
        return (numpy.max(norm_displacement), )
    elif metric == "mean":
        return (numpy.mean(norm_displacement), )
    elif metric == "stat":
        return (numpy.mean(norm_displacement), numpy.std(norm_displacement))
    else:
        raise ValueError("Unkown metric.")


def deformation_field(rigid, shape, spacing, origin):
    """ Evaluate the deformation field associated to a rigid transform.

    Parameters
    ----------
    rigid: array (4, 4)
        a rigid transformation.
    shape, sapcing, origin: array (3,)
        the image shape, spacing and origin.
    metric: str (optional, default 'max')

    Returns
    -------
    deformation_field: array (shape, 3)
        the deformation field associated with the rigid transformation.
    """
    # Go through all the image voxels
    x = numpy.arange(0, shape[0] * spacing[0], spacing[0])
    y = numpy.arange(0, shape[1] * spacing[1], spacing[1])
    z = numpy.arange(0, shape[2] * spacing[2], spacing[2])
    mesh = numpy.meshgrid(x, y, z)
    for item in mesh:
        item.shape += (1, )
    mesh = numpy.concatenate(mesh , axis=3)
            
    # Apply the rigid transform
    points = field_add(mesh, -numpy.asarray(origin))
    wrap_points = field_add(field_dot(rigid[:3, :3], points), rigid[:3, 3])
    deformation_field = wrap_points - points

    return deformation_field


def field_dot(matrix, field):
    """ Dot product between a rotation matrix and a field of 3d vectors.

    Parameters
    ----------
    matrix: array (3, 3)
        a rotation matrix.
    field: array (x, y, z, 3)
        an image of vectors to rotate.

    Returns
    -------
    dot: array (x, y, z, 3)
        the rotated field.
    """
    dot = numpy.zeros(field.shape, dtype=numpy.single)
    dot[..., 0] = numpy.sum(matrix[0, i] * field[..., i] for i in range(3))
    dot[..., 1] = numpy.sum(matrix[1, i] * field[..., i] for i in range(3))
    dot[..., 2] = numpy.sum(matrix[2, i] * field[..., i] for i in range(3))
    return dot


def field_add(field, vector):
    """ Add the vector to the field.

    Parameters
    ----------
    field: array (x, y, z, 3)
        an image of vectors.
    vector: array (3, )
        the vector that will be added to the field.

    Returns
    -------
    field: array (x, y, z, 3)
        the incremented image of vectors.
    """
    field[..., 0] += vector[0]
    field[..., 1] += vector[1]
    field[..., 2] += vector[2]
    return field



if __name__ == "__main__":

    # Z-rotation of alpha + translation
    alpha = numpy.pi / 2
    trans = [0, 0, 0]
    rigid = numpy.array([
        [numpy.cos(alpha), -numpy.sin(alpha), 0, trans[0]],
        [numpy.sin(alpha), numpy.cos(alpha),0, trans[1]],
        [0, 0, 1, trans[2]],
        [0, 0, 0, 1]
    ],dtype=numpy.single)

    # Compute the dispalcement
    dispalcement = deformation_field(rigid, (2, 2, 2), (1, 1, 1), (0, 0, 0))
    print dispalcement

    # Compute the mq
    mq = mq(rigid, (2, 2, 2), (1, 1, 1), (0, 0, 0), "stat")
    print mq   
                     
