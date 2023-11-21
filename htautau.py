# coding: utf-8

"""
Regression model for reconstructing the full di-tau system of a Higgs boson decay in the context
of the resonant X -> HH -> bb tautau search with the CMS experiment.

Contacts:
    - Tobias Kramer
    - Marcel Rieger

Code:
    - https://gist.github.com/riga/b2a8b3e1e7bbbeb390830f48405707c5
"""

from __future__ import annotations

import os
import pickle
from typing import Sequence

import numpy as np
import torch
from lumin.utils.misc import to_device


class HTauTauRegression(torch.nn.Module):
    """
    Regression model for reconstructing the full di-tau system of a Higgs or Z boson decay in the
    context of the resonant X -> HH -> bbtautau search with the CMS experiment.

    The model performs two tasks. It regresses the three-momentum components of the two emerging,
    yet undetectable neutrinos of the H -> tautau decay and simultaneously produces a classification
    output describing the compatibility of the event with the signal- or various background
    hypothesis in a multi-class classification.

    See :py:meth:`forward` for more info on the produced outputs.

    *model_file* should be a pickle'd file containing configurations and weights of the model that
    was previously pre-trained by the CMS group at University of Hamburg.
    """

    # names and numbers of custom outputs potentially produced by this model
    CUSTOM_OUTPUTS = [
        ("htt_mass", 1),
        ("htt_pt", 1),
        ("htt_eta", 1),
        ("htt_gamma", 1),
        ("htt_cos_phi", 1),
        ("hh_mass", 1),
        ("hh_cos_phi", 1),
    ]

    def __init__(
        self,
        model_file: str,
        model_version: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # load the model file
        model_file = os.path.expandvars(os.path.expanduser(model_file))
        with open(model_file, "rb") as f:
            self.m = pickle.load(f, encoding="latin1")

        # apply version-related changes
        # (none implemented so far)
        if model_version != 1:
            raise ValueError(f"unknown model version '{model_version}'")

        # setup
        self.setup_model()

    def setup_model(self) -> None:
        """
        Sets up the model.
        """
        # number of steps / batches seen during model training
        n_training_steps = int(self.m.get("training_steps", 0))

        # variables for input vector rotation
        self.rotation_anchor_name, self.rotation_names = self.m.get("rotate_phi") or (None, None)

        # reverse mapping from continuous feature names to indices in two variants:
        #   - the expected input order
        #   - the order after potential input processing (done by this module, e.g. rotate_vectors)
        self.cont_indices = {
            name: i
            for i, name in enumerate(self.m["cont_features"])
        }
        self.cont_indices_preprocessed = (
            self.cont_indices.copy()
            if not self.rotation_anchor_name else
            {
                name: i
                for i, name in enumerate(
                    name
                    for name in self.m["cont_features"]
                    if name != f"{self.rotation_anchor_name}_py"
                )
            }
        )

        # scaling values to switch between numerical domains
        self.x_cont_mean = to_device(torch.from_numpy(self.m["input_mean"]))
        self.x_cont_std = to_device(torch.from_numpy(self.m["input_variance"] ** 0.5))
        self.y_regression_mean = to_device(torch.from_numpy(self.m["regression_output_mean"]))
        self.y_regression_std = to_device(torch.from_numpy(self.m["regression_output_std"]))

        #self.x_cont_mean = torch.from_numpy(self.m["input_mean"])
        #self.x_cont_std = torch.from_numpy(self.m["input_variance"] ** 0.5)
        #self.y_regression_mean = torch.from_numpy(self.m["regression_output_mean"])
        #self.y_regression_std = torch.from_numpy(self.m["regression_output_std"])
        # helper to assign weights
        def assign_weights(tensor, attr, values, transpose=False, param=True):
            no_value = object()
            if getattr(tensor, attr, no_value) == no_value:
                raise AttributeError(f"tensor {tensor} has not attribute '{attr}'")
            if isinstance(values, str):
                values = self.m[values]
            if transpose:
                values = values.T
            values = torch.from_numpy(values)
            if param:
                values = torch.nn.Parameter(values)
            setattr(tensor, attr, values)

        # embedding for categorical features
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.m["embedding_weight"].shape[1],
            embedding_dim=self.m["embedding_weight"].shape[2],
        )
        assign_weights(self.embedding, "weight", self.m["embedding_weight"][0])

        # objects for translating embedding values to indices
        self.embedding_choices = to_device([
            torch.IntTensor(choices)
            for choices in self.m["embedding_choices"]
        ])
        #self.embedding_choices = [
        #    torch.IntTensor(choices)
        #    for choices in self.m["embedding_choices"]
        #]
        self.embedding_offsets = [
            sum(len(x) for x in self.embedding_choices[:i])
            for i in range(len(self.embedding_choices))
        ]

        # common layers
        self.common_layers = []
        self.common_batchnorms = []
        self.common_activations = []
        for i in range(1, 101):
            if f"common_{i}_weight" not in self.m:
                break

            # layer
            layer = torch.nn.Linear(*self.m[f"common_{i}_weight"].shape)
            assign_weights(layer, "weight", self.m[f"common_{i}_weight"], transpose=True)
            assign_weights(layer, "bias", self.m[f"common_{i}_bias"])
            self.common_layers.append(layer)
            setattr(self, f"common_{i}_linear", layer)

            # batch norm
            bn = torch.nn.Identity()
            if f"common_{i}_bn_mean" in self.m:
                bn = torch.nn.BatchNorm1d(
                    layer.weight.shape[0],
                    eps=float(self.m[f"common_{i}_bn_epsilon"]),
                )
                assign_weights(bn, "running_mean", f"common_{i}_bn_mean", param=False)
                assign_weights(bn, "running_var", self.m[f"common_{i}_bn_variance"], param=False)
                assign_weights(bn, "weight", f"common_{i}_bn_gamma")
                assign_weights(bn, "bias", f"common_{i}_bn_beta")
                bn.num_batches_tracked[...] = n_training_steps
                setattr(self, f"common_{i}_batchnorm", bn)
            self.common_batchnorms.append(bn)

            # activation
            act = torch.nn.ELU()
            self.common_activations.append(act)
            setattr(self, f"common_{i}_activation", act)
        else:
            raise NotImplementedError(f"more than {i} common layers detected")

        # classification layers
        self.classification_layers = []
        self.classification_batchnorms = []
        self.classification_activations = []
        for i in range(1, 101):
            is_last = False
            if f"classification_{i}_weight" not in self.m:
                i = "output"
                is_last = True

            if f"classification_{i}_weight" not in self.m:
                break

            # layer
            layer = torch.nn.Linear(*self.m[f"classification_{i}_weight"].shape)
            assign_weights(layer, "weight", self.m[f"classification_{i}_weight"], transpose=True)
            assign_weights(layer, "bias", self.m[f"classification_{i}_bias"])
            self.classification_layers.append(layer)
            setattr(self, f"classification_{i}_linear", layer)

            # batch norm
            bn = torch.nn.Identity()
            if f"classification_{i}_bn_mean" in self.m:
                bn = torch.nn.BatchNorm1d(
                    layer.weight.shape[0],
                    eps=float(self.m[f"classification_{i}_bn_epsilon"]),
                )
                assign_weights(bn, "running_mean", f"classification_{i}_bn_mean", param=False)
                assign_weights(bn, "running_var", self.m[f"classification_{i}_bn_variance"], param=False)
                assign_weights(bn, "weight", f"classification_{i}_bn_gamma")
                assign_weights(bn, "bias", f"classification_{i}_bn_beta")
                bn.num_batches_tracked[...] = n_training_steps
                setattr(self, f"classification_{i}_batchnorm", bn)
            self.classification_batchnorms.append(bn)

            # activation
            act = torch.nn.Softmax(dim=1) if is_last else torch.nn.ELU()
            self.classification_activations.append(act)
            setattr(self, f"classification_{i}_activation", act)

            # stop after the last layer
            if is_last:
                break
        else:
            raise NotImplementedError(f"more than {i} classification layers detected")

        # regression layers
        self.regression_layers = []
        self.regression_batchnorms = []
        self.regression_activations = []
        for i in range(1, 101):
            is_last = False
            if f"regression_{i}_weight" not in self.m:
                i = "output"
                is_last = True

            if f"regression_{i}_weight" not in self.m:
                break

            # layer
            layer = torch.nn.Linear(*self.m[f"regression_{i}_weight"].shape)
            assign_weights(layer, "weight", self.m[f"regression_{i}_weight"], transpose=True)
            assign_weights(layer, "bias", self.m[f"regression_{i}_bias"])
            self.regression_layers.append(layer)
            setattr(self, f"regression_{i}_linear", layer)

            # batch norm
            bn = torch.nn.Identity()
            if f"regression_{i}_bn_mean" in self.m:
                bn = torch.nn.BatchNorm1d(
                    layer.weight.shape[0],
                    eps=float(self.m[f"regression_{i}_bn_epsilon"]),
                )
                assign_weights(bn, "running_mean", f"regression_{i}_bn_mean", param=False)
                assign_weights(bn, "running_var", self.m[f"regression_{i}_bn_variance"], param=False)
                assign_weights(bn, "weight", f"regression_{i}_bn_gamma")
                assign_weights(bn, "bias", f"regression_{i}_bn_beta")
                bn.num_batches_tracked[...] = n_training_steps
                setattr(self, f"regression_{i}_batchnorm", bn)
            self.regression_batchnorms.append(bn)

            # activation
            act = torch.nn.Identity() if is_last else torch.nn.ELU()
            self.regression_activations.append(act)
            setattr(self, f"regression_{i}_activation", act)

            # stop after the last layer
            if is_last:
                break
        else:
            raise NotImplementedError(f"more than {i} regression layers detected")

    def forward(
        self,
        x_cont_hep: torch.FloatTensor,
        x_cat: torch.IntTensor,
        return_last_layers: bool = False,
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        """
        Forward pass.

        Please note that throughout the forward pass there is the disctinction between two numerical
        domains:
            - HEP domain: values as usually present in input data, e.g. in units of GeV
            - NN domain : scaled values, i.e., centered around with a reasonable small width
        This method takes scaling values stored in the ``model_file`` and automacally converts
        between if necessary. See the input and output description below for more info.

        Inputs:
            - x_cont_hep: continuous feautures in numerical HEP domain
            - x_cat: categorical features
            - return_last_layers: when True, the activated outputs of the last layer of both the
              classification and regression heads are returned as well

        Outputs:
            - classification result from softmax
            - regression results (neutrino three-momenta) in numerical NN domain
            - regression results (neutrino three-momenta) in numerical HEP domain
            - custom outputs (e.g. Htautau mass and pt based on regression) in numerical NN domain
            - custom outputs (e.g. Htautau mass and pt based on regression) in numerical HEP domain
            - activated last layer of the classification head (only of return_last_layers is True)
            - activated last layer of the regression head (only of return_last_layers is True)
        """
        # rotate particles within inputs
        if self.rotation_anchor_name:
            x_cont_hep = self.rotate_vectors(
                x_cont_hep,
                self.rotation_anchor_name,
                self.rotation_names,
            )

        x_cat = to_device(x_cat)
        x_cont_hep = to_device(x_cont_hep) 
        self.x_cont_mean = to_device(self.x_cont_mean) 
        self.x_cont_std = to_device(self.x_cont_std) 
        # scale continuous inputs from hep to nn domain
        x_cont = (x_cont_hep - self.x_cont_mean) / self.x_cont_std

        # convert embedding values to indices
        x_cat_indices = []
        for i in range(x_cat.shape[1]):
            x_cat_indices.append(
                torch.where(x_cat[:, i, None] == self.embedding_choices[i])[1] +
                self.embedding_offsets[i],
            )
        #x_cat_indices = to_device(torch.concat(
        #    list(i[..., None] for i in x_cat_indices),
        #    axis=-1,
        #))

        x_cat_indices = torch.concat(
            list(i[..., None] for i in x_cat_indices),
            axis=-1,
        )
        # apply the embedding
        x_cat_embedded = torch.flatten(self.embedding(x_cat_indices), start_dim=1)

        # concat inputs
        x = torch.concat([x_cont, x_cat_embedded], dim=1)

        # common layers
        y_common = x
        for layer, bn, act in zip(
            self.common_layers,
            self.common_batchnorms,
            self.common_activations,
        ):
            y_common = act(bn(layer(y_common)))

        # classification layers
        y_classification_prev = None
        y_classification = y_common
        for layer, bn, act in zip(
            self.classification_layers,
            self.classification_batchnorms,
            self.classification_activations,
        ):
            y_classification_prev = y_classification
            y_classification = act(bn(layer(y_classification)))

        # regression layers
        y_regression_prev = None
        y_regression = y_common
        for layer, bn, act in zip(
            self.regression_layers,
            self.regression_batchnorms,
            self.regression_activations,
        ):
            y_regression_prev = y_regression
            y_regression = act(bn(layer(y_regression)))

        # scale regression outputs from nn to hep domain
        y_regression_hep = y_regression * self.y_regression_std + self.y_regression_mean

        # additional features based on the regression output
        y_custom, y_custom_hep = self.build_custom_outputs(
            x_cont_hep,
            y_regression_hep,
            specs=self.m["custom_outputs"],
        )

        # prepare the tensors to output
        output = (y_classification, y_regression, y_regression_hep, y_custom, y_custom_hep)
        if return_last_layers:
            output += (y_classification_prev, y_regression_prev)

        return output

    @property
    def input_names(self) -> tuple[list[str], list[str]]:
        """
        Returns the list of names of expected continuous and categorical input feautres in a
        2-tuple.
        """
        return (self.m["cont_features"], self.m["cat_features"])

    def rotate_vectors(
        self,
        x_cont_hep: torch.FloatTensor,
        anchor_name: str,
        rotate_names: Sequence[str],
    ) -> torch.FloatTensor:
        """
        Takes the continuous inputs *x_cont_hep* and rotates three-vectors of selected particles
        named *rotate_names* in the direction of the particle denote by *anchor_name* (by accessing
        vector components ``*_px`` and ``*_py``).

        After the rotation, the py component of the anchor particle is always zero and will be
        dropped from the returned tensor.
        """
        # trivial case
        if not rotate_names:
            return x_cont_hep

        # store input features in a list for later concatenation
        x = [x_cont_hep[:, i] for i in range(x_cont_hep.shape[1])]

        # get phi of the anchor to rotate against
        anchor_phi = torch.atan2(
            x[self.cont_indices[f"{anchor_name}_py"]],
            x[self.cont_indices[f"{anchor_name}_px"]],
        )

        # rotate particles
        for name in rotate_names:
            # compute pt
            px = x[self.cont_indices[f"{name}_px"]]
            py = x[self.cont_indices[f"{name}_py"]]
            pt = (px**2.0 + py**2.0)**0.5

            # compute the new phi angle
            phi_new = torch.atan2(py, px) - anchor_phi

            # compute new momentum components
            px_new = pt * torch.cos(phi_new)
            py_new = pt * torch.sin(phi_new)

            # inject new values
            x[self.cont_indices[f"{name}_px"]] = px_new
            x[self.cont_indices[f"{name}_py"]] = py_new

        # concatenate features again, dropping the py value of the anchor which would always be 0
        x_cont_hep = torch.concat(
            [
                v[..., None]
                for i, v in enumerate(x)
                if i != self.cont_indices[f"{anchor_name}_py"]
            ],
            dim=1,
        )

        return x_cont_hep

    def build_custom_outputs(
        self,
        x_cont_hep: torch.FloatTensor,
        y_regression_hep: torch.FloatTensor,
        specs: dict[str, Sequence[float]],
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Takes the continuous inputs *x_cont_hep* and the regression output *y_regression_hep* (both
        in numerical HEP domain) and returns custom features that might be useful for subsequent
        evaluation or training steps.

        The produced outputs depend on *specs* which should be a dictionary mapping output names to
        sequences of length 2, containing mean and standard deviation values for subsequent scaling
        from numerical HEP to NN domains.
        """
        vector_components = ["e", "px", "py", "pz"]
        #to_device(self.cont_indices_preprocessed)

        # build visible taus (daus) from continuous inputs
        dau1, dau2 = [
            torch.concat(
                [
                    x_cont_hep[:, self.cont_indices_preprocessed[f"dau{i + 1}_{c}"]][..., None]
                    for c in vector_components
                ],
                axis=1,
            )
            for i in range(2)
        ]

        # build neutrinos from up-scaled regression outputs
        nu1, nu2 = [
            torch.concat(
                [
                    y_regression_hep[:, 3 * i + j][..., None]
                    for j in range(3)
                ],
                axis=1,
            )
            for i in range(2)
        ]
        # add leading energy components
        nu1 = torch.concat([
            ((nu1[:, 0]**2.0 + nu1[:, 1]**2.0 + nu1[:, 2]**2.0)**0.5)[..., None],
            nu1,
        ], axis=1)
        nu2 = torch.concat([
            ((nu2[:, 0]**2.0 + nu2[:, 1]**2.0 + nu2[:, 2]**2.0)**0.5)[..., None],
            nu2,
        ], axis=1)

        # htt candidate
        tau1 = dau1 + nu1
        tau2 = dau2 + nu2
        htt = tau1 + tau2

        # start building variables, depending on the "custom_outputs" content
        custom_outputs_hep = []
        custom_outputs = []

        # helper to add outputs
        def add_output(name: str, output: torch.FloatTensor) -> None:
            # add an axis for later concatenation
            if len(output.shape) < 3:
                output = output[..., None]
            # add the output
            custom_outputs_hep.append(output)
            # add the nn domain scaled output
            mean, std = specs[name]
            custom_outputs.append((custom_outputs_hep[-1] - mean) / std)

        # helper to check whether an output should be produced
        def need_output(name: str) -> bool:
            return specs.get(name) is not None

        # htt_mass
        if need_output("htt_mass") or need_output("htt_gamma"):
            htt_mass = (htt[:, 0]**2.0 - htt[:, 1]**2.0 - htt[:, 2]**2.0 - htt[:, 3]**2.0)**0.5
        if need_output("htt_mass"):
            add_output("htt_mass", htt_mass)

        # htt_pt
        if need_output("htt_pt"):
            htt_pt = (htt[:, 1]**2.0 + htt[:, 2]**2.0)**0.5
            add_output("htt_pt", htt_pt)

        # htt_eta
        if need_output("htt_eta"):
            htt_eta = torch.asinh(htt[:, 3] / htt_pt)
            add_output("htt_eta", htt_eta)

        # htt_gamma
        if need_output("htt_gamma"):
            htt_gamma = htt[:, 0] / htt_mass
            add_output("htt_gamma", htt_gamma)

        # htt_cos_phi
        if need_output("htt_cos_phi"):
            htt_cos_phi = (
                (tau1[:, 1] * tau2[:, 1] + tau1[:, 2] * tau2[:, 2] + tau1[:, 3] * tau2[:, 3]) /
                (tau1[:, 1]**2.0 + tau1[:, 2]**2.0 + tau1[:, 3]**2.0)**0.5 /
                (tau2[:, 1]**2.0 + tau2[:, 2]**2.0 + tau2[:, 3]**2.0)**0.5
            )
            add_output("htt_cos_phi", htt_cos_phi)

        # hh_mass
        if need_output("hh_mass") or need_output("hh_cos_phi"):
            b1, b2 = [
                torch.concat(
                    [
                        x_cont_hep[:, self.cont_indices_preprocessed[f"bjet{i + 1}_{v}"]][..., None]
                        for v in vector_components
                    ],
                    axis=1,
                )
                for i in range(2)
            ]
            hbb = b1 + b2
        if need_output("hh_mass"):
            hh = htt + hbb
            hh_mass = (hh[:, 0]**2.0 - hh[:, 1]**2.0 - hh[:, 2]**2.0 - hh[:, 3]**2.0)**0.5
            add_output("hh_mass", hh_mass)

        # hh_cos_phi
        if need_output("hh_cos_phi"):
            hh_cos_phi = (
                (htt[:, 1] * hbb[:, 1] + htt[:, 2] * hbb[:, 2] + htt[:, 3] * hbb[:, 3]) /
                (htt[:, 1]**2.0 + htt[:, 2]**2.0 + htt[:, 3]**2.0)**0.5 /
                (hbb[:, 1]**2.0 + hbb[:, 2]**2.0 + hbb[:, 3]**2.0)**0.5
            )
            add_output("hh_cos_phi", hh_cos_phi)

        # concatenate outputs
        custom_outputs_hep = torch.concat(custom_outputs_hep, dim=1)
        custom_outputs = torch.concat(custom_outputs, dim=1)

        return custom_outputs, custom_outputs_hep

    @property
    def custom_output_names(self) -> list[str]:
        """
        Returns a list of names of custom outputs that are produced by this model, depending on the
        content of the provided model file. Logically, this will be a subset of
        :py:attr:`CUSTOM_OUTPUTS`.
        """
        return [
            name
            for name, _ in self.CUSTOM_OUTPUTS
            if self.m["custom_outputs"].get(name)
        ]

    @property
    def output_shapes(self) -> tuple[int, int, int, int, int]:
        """
        Returns a tuple with the number of output features of each tensor created by the forward
        pass.
        """
        # get shapes
        n_cls = self.m["classification_output_weight"].shape[1]
        n_reg = self.m["regression_output_weight"].shape[1]
        n_custom = sum(
            n
            for name, n in self.CUSTOM_OUTPUTS
            if self.m["custom_outputs"].get(name)
        )

        return (n_cls, n_reg, n_reg, n_custom, n_custom)

    def train(self, mode: bool = True) -> "HTauTauRegression":
        """
        Sets this layer and all its sublayers to training (evaluation) mode if *mode* is *True*
        (*False*).
        """
        ret = super().train(mode)

        # forward to all layers that need it
        for bn in self.common_batchnorms + self.regression_batchnorms + self.classification_batchnorms:
            if bn.__class__.__name__ == "BatchNorm1d":
                bn.train(mode)

        return ret

    def eval(self) -> "HTauTauRegression":
        """
        Shorthand for ``self.train(mode=False)``.
        """
        return self.train(mode=False)


# testing
if __name__ == "__main__":
    # create a model
    model = HTauTauRegression(model_file="model4.pkl")
    model.eval()
    print(model)

    # print input feature names
    names = model.input_names
    print(f"continuous inputs : {','.join(names[0])}\n")
    print(f"categorical inputs: {','.join(names[1])}\n")

    # test data
    x_cont_hep = torch.from_numpy(np.array(
        [[
            164.3132781982422,
            159.3338165283203,
            151.51536560058594,
            221.88299560546875,
            134.232177734375,
            186.0080108642578,
            0.7036625146865845,
            0.2281920313835144,
            25.49605941772461,
            24.88109016418457,
            -36.07587432861328,
            50.70095443725586,
            0.015081183053553104,
            0.02132377400994301,
            0.0,
            2.519103527069092,
            26.736488342285156,
            -36.70349884033203,
            45.4791374206543,
            0.014414062723517418,
            0.0075927735306322575,
            0.9689258933067322,
            1367.73974609375,
            105.70636749267578,
            1256.5947265625,
            9.74587631225586,
            -55.11505889892578,
            -70.44725799560547,
            90.40850830078125,
            0.9534387588500977,
            0.03933991864323616,
            0.0029190967325121164,
            0.0011505824513733387,
            0.9558165073394775,
            0.013624761253595352,
            0.008891228586435318,
            0.013641241006553173,
            0.003870404791086912,
            8.617871208116412e-05,
            0.779367983341217,
            -18.35356903076172,
            -111.77648162841797,
            -7.852819442749023,
            114.3111801147461,
            0.6101833581924438,
            0.2371222823858261,
            0.001905966317281127,
            0.06413445621728897,
            0.6495890617370605,
            0.19755631685256958,
            0.05119197443127632,
            0.03561791032552719,
            1.3347957406040223e-08,
            4.264187282387866e-06,
            0.7793510556221008,
        ]] * 3,
        dtype=np.float32,
    ))
    x_cat = torch.from_numpy(np.array([[1, -1, 0]] * 3, dtype=np.int32))

    # evaluate
    with torch.no_grad():
        cls_nn, reg_nn, reg_hep, custom_nn, custom_hep, *last_layers = model(
            x_cont_hep,
            x_cat,
            return_last_layers=True,
        )

    print(f"""


predidcted output:
# output:
# neutrino momenta (normalized): {[round(v, 6) for v in reg_nn.tolist()[0]]}
# neutrino momenta (GeV)       : {[round(v, 6) for v in reg_hep.tolist()[0]]}
# classification               : {[round(v, 6) for v in cls_nn.tolist()[0]]}
# custom output names          : {model.custom_output_names}
# custom output values         : {[round(v, 6) for v in custom_hep.tolist()[0]]}
    """)

    print("""


expected output:
# output:
# neutrino momenta (normalized): [1.4583696, -1.2010516, -1.5615261, 0.7330556, 2.4292517, -1.504133]
# neutrino momenta (GeV)       : [200.2516, -30.97596, -200.09364, 80.24028 , 47.360962, -119.53815]
# classification               : [0.9958528, 0.00311541, 0.00103178]
# custom output names          : ['htt_mass', 'htt_pt', 'htt_eta', 'htt_gamma', 'htt_cos_phi', 'hh_mass', 'hh_cos_phi']
# custom output values         : [119.46855, 339.58282, -0.9872149, 4.457393, 0.8922215, 527.3856, -0.19826911]
    """)

    if last_layers:
        cls_last, reg_last = last_layers
        print(f"""


    output of last layers:
    classification: shape: {cls_last.shape}, first elements: {cls_last[0, 0]}, {cls_last[0, 1]}, {cls_last[0, 2]} ...
    regression    : shape: {reg_last.shape}, first elements: {reg_last[0, 0]}, {reg_last[0, 1]}, {reg_last[0, 2]} ...
        """)
