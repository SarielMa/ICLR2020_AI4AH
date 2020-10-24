import numpy as np
import torch
from torch import optim
import torch.nn.functional as nnF
from Evaluate import clip_norm_, normalize_grad_
#%%
"""
#https://github.com/cassidylaidlaw/cleverhans/blob/master/cleverhans/future/torch/attacks/spsa.py
# https://arxiv.org/pdf/1802.05666.pdf
The SPSA attack."""

def clip_eta(eta, norm, eps):

  """

  PyTorch implementation of the clip_eta in utils_tf.



  :param eta: Tensor

  :param norm: np.inf, 1, or 2

  :param eps: float

  """

  if norm not in [np.inf, 1, 2]:

    raise ValueError('norm must be np.inf, 1, or 2.')



  avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)

  reduc_ind = list(range(1, len(eta.size())))

  if norm == np.inf:

    eta = torch.clamp(eta, -eps, eps)

  else:

    if norm == 1:

      raise NotImplementedError("L1 clip is not implemented.")

      norm = torch.max(

          avoid_zero_div,

          torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True)

      )

    elif norm == 2:

      norm = torch.sqrt(torch.max(

          avoid_zero_div,

          torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)

      ))

    factor = torch.min(

        torch.tensor(1., dtype=eta.dtype, device=eta.device),

        eps / norm

        )

    eta *= factor

  return eta


#%%
def spsa(model_fn, x, eps, nb_iter, norm=np.inf, clip_min=-np.inf, clip_max=np.inf, y=None,

         targeted=False, early_stop_loss_threshold=None, learning_rate=0.01, delta=0.01,

         spsa_samples=128, spsa_iters=1, is_debug=False, sanity_checks=True):

  """

  This implements the SPSA adversary, as in https://arxiv.org/abs/1802.05666

  (Uesato et al. 2018). SPSA is a gradient-free optimization method, which is useful when

  the model is non-differentiable, or more generally, the gradients do not point in useful

  directions.



  :param model_fn: A callable that takes an input tensor and returns the model logits.

  :param x: Input tensor.

  :param eps: The size of the maximum perturbation, measured in the L-infinity norm.

  :param nb_iter: The number of optimization steps.

  :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.

  :param clip_min: If specified, the minimum input value.

  :param clip_max: If specified, the maximum input value.

  :param y: (optional) Tensor with true labels. If targeted is true, then provide the

            target label. Otherwise, only provide this parameter if you'd like to use true

            labels when crafting adversarial samples. Otherwise, model predictions are used

            as labels to avoid the "label leaking" effect (explained in this paper:

            https://arxiv.org/abs/1611.01236). Default is None.

  :param targeted: (optional) bool. Is the attack targeted or untargeted? Untargeted, the

            default, will try to make the label incorrect. Targeted will instead try to

            move in the direction of being more like y.

  :param early_stop_loss_threshold: A float or None. If specified, the attack will end as

            soon as the loss is below `early_stop_loss_threshold`.

  :param learning_rate: Learning rate of ADAM optimizer.

  :param delta: Perturbation size used for SPSA approximation.

  :param spsa_samples:  Number of inputs to evaluate at a single time. The true batch size

            (the number of evaluated inputs for each update) is `spsa_samples *

            spsa_iters`

  :param spsa_iters:  Number of model evaluations before performing an update, where each

            evaluation is on `spsa_samples` different inputs.

  :param is_debug: If True, print the adversarial loss after each update.

  :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /

            memory or for unit tests that intentionally pass strange input)

  :return: a tensor for the adversarial example

  """



  if y is not None and len(x) != len(y):

    raise ValueError('number of inputs {} is different from number of labels {}'

                     .format(len(x), len(y)))

  if y is None:

    y = torch.argmax(model_fn(x), dim=1)



  # The rest of the function doesn't support batches of size greater than 1,

  # so if the batch is bigger we split it up.

  if len(x) != 1:

    adv_x = []

    for x_single, y_single in zip(x, y):

      adv_x_single = spsa(model_fn=model_fn, x=x_single.unsqueeze(0), eps=eps,

                          nb_iter=nb_iter, norm=norm, clip_min=clip_min, clip_max=clip_max,

                          y=y_single.unsqueeze(0), targeted=targeted,

                          early_stop_loss_threshold=early_stop_loss_threshold,

                          learning_rate=learning_rate, delta=delta,

                          spsa_samples=spsa_samples, spsa_iters=spsa_iters,

                          is_debug=is_debug, sanity_checks=sanity_checks)

      adv_x.append(adv_x_single)

    return torch.cat(adv_x)



  if eps < 0:

    raise ValueError(

        "eps must be greater than or equal to 0, got {} instead".format(eps))

  if eps == 0:

    return x



  if clip_min is not None and clip_max is not None:

    if clip_min > clip_max:

      raise ValueError(

          "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}"

          .format(clip_min, clip_max))



  asserts = []



  # If a data range was specified, check that the input was in that range

  asserts.append(torch.all(x >= clip_min))

  asserts.append(torch.all(x <= clip_max))



  if is_debug:

    print("Starting SPSA attack with eps = {}".format(eps))



  perturbation = (torch.rand_like(x) * 2 - 1) * eps

  _project_perturbation(perturbation, norm, eps, x, clip_min, clip_max)

  optimizer = optim.Adam([perturbation], lr=learning_rate)



  for i in range(nb_iter):

    def loss_fn(pert):

      """

      Margin logit loss, with correct sign for targeted vs untargeted loss.

      """

      logits = model_fn(x + pert)

      loss_multiplier = 1 if targeted else -1

      return loss_multiplier * _margin_logit_loss(logits, y.expand(len(pert)))



    spsa_grad = _compute_spsa_gradient(loss_fn, x, delta=delta,

                                       samples=spsa_samples, iters=spsa_iters)

    perturbation.grad = spsa_grad

    optimizer.step()



    _project_perturbation(perturbation, norm, eps, x, clip_min, clip_max)



    loss = loss_fn(perturbation).item()

    if is_debug:

      print('Iteration {}: loss = {}'.format(i, loss))

    if early_stop_loss_threshold is not None and loss < early_stop_loss_threshold:

      break
    

  adv_x = torch.clamp((x + perturbation).detach(), clip_min, clip_max)



  if norm == np.inf:

    asserts.append(torch.all(torch.abs(adv_x - x) <= eps + 1e-6))

  else:

    asserts.append(torch.all(torch.abs(

      torch.renorm(adv_x - x, p=norm, dim=0,  maxnorm=eps) - (adv_x - x)) < 1e-6))

  asserts.append(torch.all(adv_x >= clip_min))

  asserts.append(torch.all(adv_x <= clip_max))



  if sanity_checks:

    assert np.all(asserts)



  return adv_x




#%%
def _project_perturbation(perturbation, norm, epsilon, input_image, clip_min=-np.inf,

                          clip_max=np.inf):

  """

  Project `perturbation` onto L-infinity ball of radius `epsilon`. Also project into

  hypercube such that the resulting adversarial example is between clip_min and clip_max,

  if applicable. This is an in-place operation.

  """



  clipped_perturbation = clip_eta(perturbation, norm, epsilon)

  new_image = torch.clamp(input_image + clipped_perturbation,

                          clip_min, clip_max)



  perturbation.add_((new_image - input_image) - perturbation)




#%%
def _compute_spsa_gradient(loss_fn, x, delta, samples, iters):

  """

  Approximately compute the gradient of `loss_fn` at `x` using SPSA with the

  given parameters. The gradient is approximated by evaluating `iters` batches

  of `samples` size each.

  """



  assert len(x) == 1

  num_dims = len(x.size())



  x_batch = x.expand(samples, *([-1] * (num_dims - 1)))



  grad_list = []

  for i in range(iters):

    delta_x = delta * torch.sign(torch.rand_like(x_batch) - 0.5)

    delta_x = torch.cat([delta_x, -delta_x])

    loss_vals = loss_fn(x + delta_x)
    
    #print('loss_vals', loss_vals.shape)

    while len(loss_vals.size()) < num_dims:

      loss_vals = loss_vals.unsqueeze(-1)

    avg_grad = torch.mean(loss_vals * torch.sign(delta_x), dim=0, keepdim=True) / delta

    grad_list.append(avg_grad)



  return torch.mean(torch.cat(grad_list), dim=0, keepdim=True)




#%%
def _margin_logit_loss(logits, labels):

  """

  Computes difference between logits for `labels` and next highest logits.



  The loss is high when `label` is unlikely (targeted by default).

  """



  correct_logits = logits.gather(1, labels[:, None]).squeeze(1)



  logit_indices = torch.arange(

      logits.size()[1],

      dtype=labels.dtype,

      device=labels.device,

  )[None, :].expand(labels.size()[0], -1)

  incorrect_logits = torch.where(

      logit_indices == labels[:, None],

      torch.full_like(logits, float('-inf')),

      logits,

  )

  max_incorrect_logits, _ = torch.max(

      incorrect_logits, 1)

  
  #print('a', logits.shape)
  #print('b', max_incorrect_logits.shape)

  return max_incorrect_logits - correct_logits
#%% no reduction
def margin_loss(Z, Y):
    if len(Z.size()) <= 1:
        t = Y*2-1
        loss=-Z*t
        loss=loss.view(-1,1)
    else:
        num_classes=Z.size(1)
        Zy=torch.gather(Z, 1, Y[:,None])
        idxTable= torch.arange(0, num_classes, dtype=torch.int64, device=Z.device).expand(Z.size())
        Zother=torch.where(idxTable!=Y[:,None], Z,  torch.full_like(Z, float('-inf')))
        Zother_max = Zother.max(dim=1, keepdim=True)[0]
        loss=Zother_max-Zy
    return loss 
#%%
def spsa_attack(model, X, Y, noise_norm, norm_type=np.inf, max_iter=100, step=0.01, lr=0.01,
                rand_init=True, rand_init_max=None, spsa_samples=128, spsa_iters=1, targeted=False, use_optimizer=True):
    model.eval()#set model to evaluation mode
    X = X.detach()    
    #-----------------
    if rand_init is True:
        init_value=rand_init_max
        if rand_init_max is None:
            init_value=noise_norm            
        noise_init=init_value*(2*torch.rand_like(X)-1)
        clip_norm_(noise_init, norm_type, noise_norm)
        Xn = X + noise_init
    else:
        Xn = X.clone().detach() # must clone
    #-----------------
    l_sign=-1
    if targeted == True:
        l_sign=1
    #-----------------
    with torch.no_grad():
        for k in range(0, X.size(0)):
            noise=Xn[k]-X[k]
            if use_optimizer == True:
                optimizer = optim.Adamax([noise], lr=lr)
            for iter in range(0, max_iter):
                grad_n=torch.zeros_like(noise)
                for s_iter in range(0, spsa_iters):
                    Xnk=Xn[k].expand((spsa_samples, *Xn[k].size()))
                    v = torch.sign(torch.rand_like(Xnk) - 0.5)
                    delta_x = step*v
                    Z1=model(Xnk+delta_x)
                    Z2=model(Xnk-delta_x)
                    Yk=Y[k].expand(spsa_samples)
                    f1=l_sign*margin_loss(Z1, Yk)
                    f2=l_sign*margin_loss(Z2, Yk)
                    v=v.view(v.size(0), -1)
                    grad_s=(f1-f2)*v/(2*step)
                    grad_s=grad_s.mean(dim=0).view(noise.shape)
                    grad_n+=grad_s
                #---------------------------------------------
                grad_n/=spsa_iters
                normalize_grad_(grad_n, norm_type)                
                if use_optimizer == True:
                    noise.grad=grad_n
                    optimizer.step()
                else:
                    noise = noise - lr*grad_n
                clip_norm_(noise.view((1, *noise.size())), norm_type, noise_norm)
                Xn[k] = torch.clamp(X[k]+noise, 0, 1)
                noise.data-= noise.data-(Xn[k]-X[k]).data
                #----------------------------
                #stop if lable change
                Xnk=Xn[k].view((1, *Xn[k].size()))
                Zk=model(Xnk)
                Yk=Y[k].to(torch.int64)
                if len(Zk.size()) <= 1:
                    Ykp = (Zk>0).to(torch.int64)                    
                else:
                    Ykp = Zk.data.max(dim=1)[1]                
                if targeted == False and Ykp != Yk:
                    #print('break, k=', k, ', iter=', iter)
                    break
                elif targeted == True and Ykp == Yk:
                    break
    return Xn
#%%
def test_adv(model, device, dataloader, num_classes,
             noise_norm, max_iter=100, step=0.01, norm_type=np.inf, targeted=False,
             method='spsa', spsa_samples=128, spsa_iters=1, max_batch=None, use_optimizer=True):
    model.eval()#set model to evaluation mode
    confusion_clean=np.zeros((num_classes,num_classes))
    confusion_noisy=np.zeros((num_classes,num_classes))
    sample_count=0
    adv_sample_count=0
    sample_idx_wrong=[]
    sample_idx_attack=[]
    #---------------------
    for batch_idx, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        #------------------
        Z = model(X)#classify the 'clean' signal X
        if len(Z.size()) <= 1:
            Yp = (Z>0).to(torch.int64) #binary/sigmoid
        else:
            Yp = Z.data.max(dim=1)[1] #multiclass/softmax            
        #------------------        
        if method == 'spsa':
            Xn=spsa(model_fn=model, x=X, y=Y, eps=noise_norm, nb_iter=max_iter, delta=step,
                    spsa_samples=spsa_samples, spsa_iters=spsa_iters,
                    early_stop_loss_threshold=-0.01, # stop if label change
                    norm=norm_type, clip_min=0, clip_max=1, targeted=targeted)     
        elif method == 'spsa_attack':
            Xn=spsa_attack(model, X, Y, noise_norm, norm_type, max_iter, step, lr=step,
                           spsa_samples=spsa_samples, spsa_iters=spsa_iters, 
                           targeted=targeted, use_optimizer=use_optimizer)
        else:
            raise NotImplementedError("other method is not implemented.")
        #------------------
        Zn = model(Xn)# classify the noisy signal Xn
        if len(Z.size()) <= 1:
            Ypn = (Zn>0).to(torch.int64)
        else:
            Ypn = Zn.data.max(dim=1)[1]
        #------------------
        if len(Z.size()) <= 1:
            Y = (Y>0.5).to(torch.int64)
        #------------------
        #do not attack x that is missclassified
        Ypn_ = Ypn.clone().detach()
        Zn_=Zn.clone().detach()
        if targeted == False:
            temp=(Yp!=Y)
            Ypn_[temp]=Yp[temp]        
            Zn_[temp]=Z[temp]
        for i in range(0, confusion_noisy.shape[0]):
            for j in range(0, confusion_noisy.shape[1]):
                confusion_noisy[i,j]+=torch.sum((Y==i)&(Ypn_==j)).item()
        #------------------
        for i in range(0, confusion_clean.shape[0]):
            for j in range(0, confusion_clean.shape[1]):
                confusion_clean[i,j]+=torch.sum((Y==i)&(Yp==j)).item()
        #------------------
        for m in range(0,X.size(0)):
            idx=sample_count+m            
            if Y[m] != Yp[m]:
                sample_idx_wrong.append(idx)
            elif Ypn[m] != Yp[m]:
                sample_idx_attack.append(idx)
        sample_count+=X.size(0)
        adv_sample_count+=torch.sum((Yp==Y)&(Ypn!=Y)).item()
        #------------------
        print('batch_idx=', batch_idx, ', acc=', (Yp==Y).sum().item()/X.size(0), 
              ', adv acc=', (Ypn==Y).sum().item()/X.size(0))
        if batch_idx % 10 == 0:
            acc_clean = confusion_clean.diagonal().sum()/confusion_clean.sum()
            acc_noisy = confusion_noisy.diagonal().sum()/confusion_noisy.sum()
            print('test_adv: {} [{:.0f}%], acc_clean: {:.3f}, acc_noisy: {:.3f}, noise_norm: {}, norm_type: {}'.format(
                  batch_idx, 100. * batch_idx / len(dataloader), acc_clean, acc_noisy, noise_norm, norm_type))
        if max_batch is not None:
            if batch_idx>= max_batch:
                print('max_batch=', max_batch, 'is reached, stop test')
                break
    #------------------
    #------------------
    acc_clean = confusion_clean.diagonal().sum()/confusion_clean.sum()
    acc_noisy = confusion_noisy.diagonal().sum()/confusion_noisy.sum()
    sens_clean=np.zeros(num_classes)
    prec_clean=np.zeros(num_classes)
    for m in range(0, num_classes):
        sens_clean[m]=confusion_clean[m,m]/np.sum(confusion_clean[m,:])
        prec_clean[m]=confusion_clean[m,m]/np.sum(confusion_clean[:,m])
    sens_noisy=np.zeros(num_classes)
    prec_noisy=np.zeros(num_classes)
    for m in range(0, num_classes):
        sens_noisy[m]=confusion_noisy[m,m]/np.sum(confusion_noisy[m,:])
        prec_noisy[m]=confusion_noisy[m,m]/np.sum(confusion_noisy[:,m])
    #------------------
    result={}
    result['method']='spsa'
    result['noise_norm']=noise_norm
    result['norm_type']=norm_type
    result['max_iter']=max_iter
    result['step']=step
    result['sample_count']=sample_count
    result['adv_sample_count']=adv_sample_count
    result['sample_idx_wrong']=sample_idx_wrong
    result['sample_idx_attack']=sample_idx_attack
    result['confusion_clean']=confusion_clean
    result['acc_clean']=acc_clean
    result['sens_clean']=sens_clean
    result['prec_clean']=prec_clean
    result['confusion_noisy']=confusion_noisy
    result['acc_noisy']=acc_noisy
    result['sens_noisy']=sens_noisy
    result['prec_noisy']=prec_noisy
    #------------------
    print('testing robustness ', method, ', adv%=', adv_sample_count/sample_count, sep='')
    print('norm_type:', norm_type, ', noise_norm:', noise_norm, ', max_iter:', max_iter, ', step:', step,
          ', spsa_samples:', spsa_samples, ', spsa_iters:', spsa_iters, sep='')    
    print('acc_clean', result['acc_clean'], ', acc_noisy', result['acc_noisy'])
    print('sens_clean', result['sens_clean'])
    print('sens_noisy', result['sens_noisy'])
    print('prec_clean', result['prec_clean'])
    print('prec_noisy', result['prec_noisy'])
    return result
#%%