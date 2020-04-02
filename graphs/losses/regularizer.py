# -*- coding: utf-8 -*-
__author__ = 'maoss2'


class _Regularizer(object):
    """
    Parent class of Regularizers
    """

    def __init__(self, model):
        super(_Regularizer, self).__init__()
        self.model = model

    def regularized_param(self, param_weights, reg_loss_function):
        raise NotImplementedError

    def regularized_all_param(self, reg_loss_function):
        raise NotImplementedError


class L1Regularizer(_Regularizer):
    """
    L1 regularized loss
    """
    def __init__(self, model, lambda_reg=0.01):
        super(L1Regularizer, self).__init__(model=model)
        self.lambda_reg = lambda_reg

    def regularized_param(self, param_weights, reg_loss_function):
        print("Params to reg: \t", param_weights)
        reg_loss_function += self.lambda_reg * L1Regularizer.__add_l1(var=param_weights)
        return reg_loss_function

    def regularized_all_param(self, reg_loss_function):
        for model_param_name, model_param_value in self.model.named_parameters():
            if model_param_name.endswith('weight'):
                reg_loss_function += self.lambda_reg * L1Regularizer.__add_l1(var=model_param_value)
        return reg_loss_function

    @staticmethod
    def __add_l1(var):
        return var.abs().sum()


class L2Regularizer(_Regularizer):
    """
       L2 regularized loss
    """
    def __init__(self, model, lambda_reg=0.001):
        super(L2Regularizer, self).__init__(model=model)
        self.lambda_reg = lambda_reg


    def regularized_param(self, param_weights, reg_loss_function):
        # print("Params to reg: \t", param_weights)
        reg_loss_function += self.lambda_reg * L2Regularizer.__add_l2(var=param_weights)
        return reg_loss_function

    def regularized_all_param(self, reg_loss_function):
        for model_param_name, model_param_value in self.model.named_parameters():
            if model_param_name.endswith('weight'):
                # print("Params to reg: \t", model_param_name)
                reg_loss_function += self.lambda_reg * L2Regularizer.__add_l2(var=model_param_value)
        return reg_loss_function

    @staticmethod
    def __add_l2(var):
        return var.pow(2).sum()


class ElasticNetRegularizer(_Regularizer):
    """
    Elastic Net Regularizer
    """
    def __init__(self, model, lambda_reg=0.01, alpha_reg=0.01):
        super(ElasticNetRegularizer, self).__init__(model=model)
        self.lambda_reg = lambda_reg
        self.alpha_reg = alpha_reg

    def regularized_param(self, param_weights, reg_loss_function):
        reg_loss_function += self.lambda_reg * \
                                     (((1 - self.alpha_reg) * ElasticNetRegularizer.__add_l2(var=param_weights)) +
                                      (self.alpha_reg * ElasticNetRegularizer.__add_l1(var=param_weights)))
        return reg_loss_function

    def regularized_all_param(self, reg_loss_function):
        for model_param_name, model_param_value in self.model.named_parameters():
            if model_param_name.endswith('weight'):
                reg_loss_function += self.lambda_reg * \
                                 (((1 - self.alpha_reg) * ElasticNetRegularizer.__add_l2(var=model_param_value)) +
                                  (self.alpha_reg * ElasticNetRegularizer.__add_l1(var=model_param_value)))
        return reg_loss_function

    @staticmethod
    def __add_l1(var):
        return var.abs().sum()

    @staticmethod
    def __add_l2(var):
        return var.pow(2).sum()

