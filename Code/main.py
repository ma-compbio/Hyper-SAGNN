from torch.nn.utils.rnn import pad_sequence
from torchsummary import summary
from gensim.models import Word2Vec
import tensorflow as tf

from scipy.sparse import csr_matrix
from scipy.sparse import vstack as s_vstack
import os
import time
import argparse
import warnings

from random_walk import random_walk
from random_walk_hyper import random_walk_hyper
from Modules import *
from utils import *

import matplotlib as mpl
mpl.use("Agg")
import multiprocessing

cpu_num = multiprocessing.cpu_count()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
device_ids = [0, 1]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings("ignore")


def parse_args():
    # Parses the node2vec arguments.
    parser = argparse.ArgumentParser(description="Run node2vec.")
    
    parser.add_argument('--data', type=str, default='ramani')
    parser.add_argument('--TRY', action='store_true')
    parser.add_argument('--FILTER', action='store_true')
    parser.add_argument('--grid', type=str, default='')
    parser.add_argument('--remark', type=str, default='')
    
    parser.add_argument('--random-walk', action='store_true')
    
    parser.add_argument('--dimensions', type=int, default=64,
                        help='Number of dimensions. Default is 64.')
    
    parser.add_argument('-l', '--walk-length', type=int, default=40,
                        help='Length of walk per source. Default is 40.')
    
    parser.add_argument('-r', '--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')
    
    parser.add_argument('-k', '--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')
    
    parser.add_argument('-i', '--iter', default=1, type=int,
                        help='Number of epochs in SGD')
    
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')
    
    parser.add_argument('--p', type=float, default=2,
                        help='Return hyperparameter. Default is 1.')
    
    parser.add_argument('--q', type=float, default=0.25,
                        help='Inout hyperparameter. Default is 1.')
    
    parser.add_argument('-a', '--alpha', type=float, default=0.0,
                        help='The weight of random walk -skip-gram loss. Default is ')
    parser.add_argument('--rw', type=float, default=0.01,
                        help='The weight of reconstruction of adjacency matrix loss. Default is ')
    parser.add_argument('-w', '--walk', type=str, default='',
                        help='The walk type, empty stands for normal rw')
    parser.add_argument('-d', '--diag', type=str, default='True',
                        help='Use the diag mask or not')
    parser.add_argument(
        '-f',
        '--feature',
        type=str,
        default='walk',
        help='Features used in the first step')
    
    args = parser.parse_args()
    
    if not args.random_walk:
        args.model_name = 'model_no_randomwalk'
        args.epoch = 25
    else:
        args.model_name = 'model_{}_'.format(args.data)
        args.epoch = 25
    if args.TRY:
        args.model_name = 'try' + args.model_name
        if not args.random_walk:
            args.epoch = 5
        else:
            args.epoch = 1
    # args.epoch = 1
    args.model_name += args.remark
    print(args.model_name)
    
    args.save_path = os.path.join(
        '../checkpoints/', args.data, args.model_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    return args


def train_batch_hyperedge(model, loss_func, batch_data, batch_weight, type, y=""):
    x = batch_data
    w = batch_weight
    
    # When label is not generated, prepare the data
    if len(y) == 0:
        x, y, w = generate_negative(x, "train_dict", type, w)
        index = torch.randperm(len(x))
        x, y, w = x[index], y[index], w[index]
    
    # forward
    pred, recon_loss = model(x, return_recon = True)
    loss = loss_func(pred, y, weight=w)
    return pred, y, loss, recon_loss


def train_batch_skipgram(model, loss_func, alpha, batch_data):
    if alpha == 0:
        return torch.zeros(1).to(device)
    
    examples, labels, neg_samples = batch_data
    
    # Embeddings for examples: [batch_size, emb_dim]
    example_emb = model.forward_u(examples)
    true_w, true_b = model.forward_w_b(labels)
    sampled_w, sampled_b = model.forward_w_b(neg_samples)
    
    # True logits: [batch_size, 1]
    true_logits = torch.sum(torch.mul(example_emb, true_w), dim=1) + true_b
    
    # Sampled logits: [batch_size, num_sampled]
    # We replicate sampled noise labels for all examples in the batch
    # using the matmul.
    sampled_b_vec = sampled_b.view(1, -1)
    
    sampled_logits = torch.matmul(example_emb,
                                  sampled_w.transpose(1, 0))
    sampled_logits += sampled_b_vec
    
    true_xent = loss_func(true_logits, torch.ones_like(true_logits).to(device))
    sampled_xent = loss_func(sampled_logits,
                             torch.zeros_like(sampled_logits).to(device))
    
    # NCE-loss is the sum of the true and noise (sampled words)
    # contributions, averaged over the batch.
    loss = (true_xent + sampled_xent) / len(examples) / len(labels)
    return loss


def train_epoch(args, model, loss_func, training_data, optimizer, batch_size, only_rw, type):
    # Epoch operation in training phase
    # Simultaneously train on 2 models: hyperedge-prediction (1) & random-walk with skipgram (2)
    model_1, model_2 = model
    (loss_1, beta), (loss_2, alpha) = loss_func
    edges, edge_weight, sentences = training_data
    y = torch.tensor([])

    
    # Permutate all the data
    index = torch.randperm(len(edges))
    edges, edge_weight = edges[index], edge_weight[index]
    if len(y) > 0:
        y = y[index]
    
    model_1.train()
    model_2.train()
    
    bce_total_loss = 0
    skipgram_total_loss = 0
    recon_total_loss = 0
    acc_list, y_list, pred_list = [], [], []
    
    batch_num = int(math.floor(len(edges) / batch_size))
    bar = trange(batch_num, mininterval=0.1, desc='  - (Training) ', leave=False, )
    for i in bar:
        if only_rw or alpha > 0:
            examples, labels, neg_samples, epoch_finished, words = sentences.next_batch()
            examples = torch.tensor(examples, dtype=torch.long, device=device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            neg_samples = torch.tensor(neg_samples, dtype=torch.long, device=device)
            loss_skipgram = train_batch_skipgram(
                model_2, loss_2, alpha, [
                    examples, labels, neg_samples])
            loss = loss_skipgram
            acc_list.append(0)
            auc1, auc2 = 0.0, 0.0
            
        else:
            batch_edge = edges[i * batch_size:(i + 1) * batch_size]
            batch_edge_weight = edge_weight[i * batch_size:(i + 1) * batch_size]
            batch_y = ""
            if len(y) > 0:
                batch_y = y[i * batch_size:(i + 1) * batch_size]
                if len(batch_y) == 0:
                    continue
            
            pred, batch_y, loss_bce, loss_recon = train_batch_hyperedge(model_1, loss_1, batch_edge, batch_edge_weight, type, y=batch_y)
            loss_skipgram = torch.Tensor([0.0]).to(device)
            loss = beta * loss_bce + alpha * loss_skipgram + loss_recon * args.rw
            acc_list.append(accuracy(pred, batch_y))
            y_list.append(batch_y)
            pred_list.append(pred)
        
        for opt in optimizer:
            opt.zero_grad()
        
        # backward
        loss.backward()
        
        # update parameters
        for opt in optimizer:
            opt.step()
        
        bar.set_description(" - (Training) BCE:  %.4f  skipgram: %.4f recon: %.4f" %
                            (bce_total_loss / (i + 1), skipgram_total_loss / (i + 1), recon_total_loss / (i + 1)))
        bce_total_loss += loss_bce.item()
        skipgram_total_loss += loss_skipgram.item()
        recon_total_loss += loss_recon.item()
    y = torch.cat(y_list)
    pred = torch.cat(pred_list)
    auc1, auc2 = roc_auc_cuda(y, pred)
    return bce_total_loss / batch_num, skipgram_total_loss / batch_num,recon_total_loss / batch_num, np.mean(acc_list), auc1, auc2


def eval_epoch(args, model, loss_func, validation_data, batch_size, type):
    ''' Epoch operation in evaluation phase '''
    bce_total_loss = 0
    recon_total_loss = 0
    (loss_1, beta), (loss_2, alpha) = loss_func
    
    loss_func = loss_1
    
    model.eval()
    with torch.no_grad():
        validation_data, validation_weight = validation_data
        y = ""
        
        index = torch.randperm(len(validation_data))
        validation_data, validation_weight = validation_data[index], validation_weight[index]
        if len(y) > 0:
            y = y[index]
        
        pred, label = [], []
        
        for i in tqdm(range(int(math.floor(len(validation_data) / batch_size))),
                      mininterval=0.1, desc='  - (Validation)   ', leave=False):
            # prepare data
            batch_x = validation_data[i * batch_size:(i + 1) * batch_size]
            batch_w = validation_weight[i * batch_size:(i + 1) * batch_size]
            
            if len(y) == 0:
                batch_x, batch_y, batch_w = generate_negative(
                    batch_x, "test_dict", type, weight=batch_w)
            else:
                batch_y = y[i * batch_size:(i + 1) * batch_size]
            
            index = torch.randperm(len(batch_x))
            batch_x, batch_y, batch_w = batch_x[index], batch_y[index], batch_w[index]
            
            pred_batch, recon_loss = model(batch_x, return_recon = True)
            pred.append(pred_batch)
            label.append(batch_y)
            
            loss = loss_func(pred_batch, batch_y, weight=batch_w)
            recon_total_loss += recon_loss.item()
            bce_total_loss += loss.item()
        
        pred = torch.cat(pred, dim=0)
        label = torch.cat(label, dim=0)
        
        acc = accuracy(pred, label)
        
        auc1, auc2 = roc_auc_cuda(label, pred)
    
    return bce_total_loss / (i + 1), recon_total_loss / (i + 1), acc, auc1, auc2


def train(args, model, loss, training_data, validation_data, optimizer, epochs, batch_size, only_rw):
    valid_accus = [0]
    # outlier_data = generate_outlier()
    
    for epoch_i in range(epochs):
        if only_rw:
            save_embeddings(model[0], True)
                
                
        
        print('[ Epoch', epoch_i, 'of', epochs, ']')
        
        start = time.time()
        
        bce_loss, skipgram_loss,recon_loss, train_accu, auc1, auc2 = train_epoch(
            args, model, loss, training_data, optimizer, batch_size, only_rw, train_type)
        print('  - (Training)   bce: {bce_loss: 7.4f}, skipgram: {skipgram_loss: 7.4f}, '
              'recon: {recon_loss: 7.4f}'
              ' acc: {accu:3.3f} %, auc: {auc1:3.3f}, aupr: {auc2:3.3f}, '
              'elapse: {elapse:3.3f} s'.format(
            bce_loss=bce_loss,
            skipgram_loss=skipgram_loss,
            recon_loss = recon_loss,
            accu=100 *
                 train_accu,
            auc1=auc1,
            auc2=auc2,
            elapse=(time.time() - start)))
        
        start = time.time()
        valid_bce_loss, recon_loss, valid_accu, valid_auc1, valid_auc2 = eval_epoch(args, model[0], loss, validation_data, batch_size,
                                                                        'hyper')
        print('  - (Validation-hyper) bce: {bce_loss: 7.4f}, recon: {recon_loss: 7.4f},'
              '  acc: {accu:3.3f} %,'
              ' auc: {auc1:3.3f}, aupr: {auc2:3.3f},'
              'elapse: {elapse:3.3f} s'.format(
            bce_loss=valid_bce_loss,
            recon_loss=recon_loss,
            accu=100 *
                 valid_accu,
            auc1=valid_auc1,
            auc2=valid_auc2,
            elapse=(time.time() - start)))
        
        valid_accus += [valid_auc1]
        # check_outlier(model[0], outlier_data)
        
        checkpoint = {
            'model_link': model[0].state_dict(),
            'model_node2vec': model[1].state_dict(),
            'epoch': epoch_i}
        
        model_name = 'model.chkpt'
        
        if valid_auc1 >= max(valid_accus):
            torch.save(checkpoint, os.path.join(args.save_path, model_name))
        
        torch.cuda.empty_cache()
        
    if not only_rw:
        checkpoint = torch.load(os.path.join(args.save_path, model_name))
        model[0].load_state_dict(checkpoint['model_link'])
        model[1].load_state_dict(checkpoint['model_node2vec'])

def generate_negative(x, dict1, get_type='all', weight="", forward=True):
    if dict1 == 'train_dict':
        dict1 = train_dict
    elif dict1 == 'test_dict':
        dict1 = test_dict
    
    if len(weight) == 0:
        weight = torch.ones(len(x), dtype=torch.float)
    
    neg_list = []
    
    zero_num_list = [0] + list(num_list)
    new_index = []
    max_id = int(num[-1])
    
    if forward:
        func1 = pass_
    else:
        func1 = tqdm
    
    if len(x.shape) > 1:
        change_list_all = np.random.randint(
            0, x.shape[-1], len(x) * neg_num).reshape((len(x), neg_num))
    for j, sample in enumerate(func1(x)):
        if len(x.shape) > 1:
            change_list = change_list_all[j, :]
        else:
            change_list = np.random.randint(0, sample.shape[-1], neg_num)
        for i in range(neg_num):
            temp = np.copy(sample)
            a = set()
            a.add(tuple(temp))
            
            trial = 0
            simple_or_hard = np.random.rand()
            if simple_or_hard <= pair_ratio:
                change = change_list[i]
                
            while not a.isdisjoint(dict1):
                temp = np.copy(sample)
                trial += 1
                if trial >= 1000:
                    temp = ""
                    break
                # Only change one node
                if simple_or_hard <= pair_ratio:
                    if len(num_list) == 1:
                        # Only one node type
                        temp[change] = np.random.randint(0, max_id, 1) + 1
                    
                    else:
                        # Multiple node types
                        start = zero_num_list[node_type_mapping[change]]
                        end = zero_num_list[node_type_mapping[change] + 1]
                        
                        temp[change] = np.random.randint(
                            int(start), int(end), 1) + 1
                else:
                    
                    if len(num_list) == 1:
                        # Only one node type
                        temp = np.random.randint(
                            0, max_id, sample.shape[-1]) + 1
                    
                    else:
                        for k in range(temp.shape[-1]):
                            start = zero_num_list[node_type_mapping[k]]
                            end = zero_num_list[node_type_mapping[k] + 1]
                            temp[k] = np.random.randint(
                                int(start), int(end), 1) + 1
                
                temp.sort()
                a = set([tuple(temp)])
            
            if len(temp) > 0:
                neg_list.append(temp)
                if i == 0:
                    new_index.append(j)
    if get_type == 'all' or get_type == 'edge':
        x_e, neg_e = generate_negative_edge(x, int(len(x)))
        if get_type == 'all':
            x = list(x) + x_e
            neg_list = neg_list + neg_e
        else:
            x = x_e
            neg_list = neg_e
    new_index = np.array(new_index)
    new_x = x[new_index]
    
    if not forward:
        device = 'cpu'
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    new_weight = torch.tensor(weight[new_index]).to(device)
    
    x = np2tensor_hyper(new_x, dtype=torch.long)
    neg = np2tensor_hyper(neg_list, dtype=torch.long)
    x = pad_sequence(x, batch_first=True, padding_value=0).to(device)
    neg = pad_sequence(neg, batch_first=True, padding_value=0).to(device)
    # print("x", x, "neg", neg)
    
    return torch.cat([x, neg]), torch.cat(
        [torch.ones((len(x), 1), device=device), torch.zeros((len(neg), 1), device=device)], dim=0), torch.cat(
        ((torch.ones((len(x), 1), device=device) * new_weight.view(-1, 1), (torch.ones((len(neg), 1), device=device)))))


def save_embeddings(model, origin=False):
    model.eval()
    with torch.no_grad():
        ids = np.arange(num_list[-1]) + 1
        ids = torch.Tensor(ids).long().to(device).view(-1, 1)
        embeddings = []
        for j in range(math.ceil(len(ids) / batch_size)):
            x = ids[j * batch_size:min((j + 1) * batch_size, len(ids))]
            if origin:
                embed = model.get_node_embeddings(x)
            else:
                embed = model.get_embedding_static(x)
            embed = embed.detach().cpu().numpy()
            embeddings.append(embed)
        
        embeddings = np.concatenate(embeddings, axis=0)[:, 0, :]
        for i in range(len(num_list)):
            start = 0 if i == 0 else num_list[i - 1]
            static = embeddings[int(start):int(num_list[i])]
            np.save("../mymodel_%d.npy" % (i), static)
            
            if origin:
                np.save("../mymodel_%d_origin.npy" % (i), static)
    
    torch.cuda.empty_cache()
    return embeddings


def generate_H(edge, nums_type, weight):
    nums_examples = len(edge)
    H = [0 for i in range(len(nums_type))]
    for i in range(edge.shape[-1]):
        # np.sqrt(weight) because the dot product later would recovers it
        H[node_type_mapping[i]] += csr_matrix((np.sqrt(weight), (edge[:, i], range(
            nums_examples))), shape=(nums_type[node_type_mapping[i]], nums_examples))
    return H


def generate_embeddings(edge, nums_type, H=None, weight=1):
    if len(num) == 1:
        return [get_adjacency(edge, True)]
    if H is None:
        H = generate_H(edge, nums_type, weight)
    
    embeddings = [H[i].dot(s_vstack([H[j] for j in range(len(num))]).T).astype('float32') for i in
                  range(len(nums_type))]
    
    new_embeddings = []
    zero_num_list = [0] + list(num_list)
    for i, e in enumerate(embeddings):
        # This is to remove diag entrance
        for j, k in enumerate(range(zero_num_list[i], zero_num_list[i + 1])):
            e[j, k] = 0
        
        # Automatically removes all zero entries
        col_sum = np.array(e.sum(0)).reshape((-1))
        new_e = e[:, col_sum > 0]
        new_e.eliminate_zeros()
        new_embeddings.append(new_e)
    
    
    # 0-1 scaling
    for i in range(len(nums_type)):
        col_max = np.array(new_embeddings[i].max(0).todense()).flatten()
        _, col_index = new_embeddings[i].nonzero()
        new_embeddings[i].data /= col_max[col_index]
    return [new_embeddings[i] for i in range(len(nums_type))]


def get_adjacency(data, norm=True):
    A = np.zeros((num_list[-1], num_list[-1]))
    
    for datum in tqdm(data):
        for i in range(datum.shape[-1]):
            for j in range(datum.shape[-1]):
                if i != j:
                    A[datum[i], datum[j]] += 1.0
    
    if norm:
        temp = np.concatenate((np.zeros((1), dtype='int'), num), axis=0)
        temp = np.cumsum(temp)
        
        for i in range(len(temp) - 1):
            A[temp[i]:temp[i + 1],
            :] /= (np.max(A[temp[i]:temp[i + 1],
                          :],
                          axis=0,
                          keepdims=True) + 1e-10)
    
    return csr_matrix(A).astype('float32')

args = parse_args()
neg_num = 5
batch_size = 96
neg_num_w2v = 5
bottle_neck = args.dimensions
pair_ratio = 0.9
train_type = 'hyper'



train_zip = np.load("../data/%s/train_data.npz" % (args.data), allow_pickle=True)
test_zip = np.load("../data/%s/test_data.npz" % (args.data), allow_pickle=True)
train_data, test_data = train_zip['train_data'], test_zip['test_data']



try:
    train_weight, test_weight = train_zip["train_weight"].astype('float32'), test_zip["test_weight"].astype('float32')
except BaseException:
    print("no specific train weight")
    test_weight = np.ones(len(test_data), dtype='float32')
    train_weight = np.ones(len(train_data), dtype='float32') * neg_num

num = train_zip['nums_type']
num_list = np.cumsum(num)
print("Node type num", num)


if len(num) > 1:
    node_type_mapping = [0, 1, 2]
    

if args.feature == 'adj':
    embeddings_initial = generate_embeddings(train_data, num, H=None, weight=train_weight)

print(train_weight)
print(train_weight, np.min(train_weight), np.max(train_weight))
train_weight_mean = np.mean(train_weight)
train_weight = train_weight / train_weight_mean * neg_num
test_weight = test_weight / train_weight_mean * neg_num



# Now for multiple node types, the first column id starts at 0, the second
# starts at num_list[0]...
if len(num) > 1:
    for i in range(len(node_type_mapping) - 1):
        train_data[:, i + 1] += num_list[node_type_mapping[i + 1] - 1]
        test_data[:, i + 1] += num_list[node_type_mapping[i + 1] - 1]

num = torch.as_tensor(num)
num_list = torch.as_tensor(num_list)

print("walk type", args.walk)
# At this stage, the index still starts from zero

node_list = np.arange(num_list[-1]).astype('int')
if args.walk == 'hyper':
    walk_path = random_walk_hyper(args, node_list, train_data)
else:
    walk_path = random_walk(args, num, train_data)
del node_list


# Add 1 for the padding index
print("adding pad idx")
train_data = add_padding_idx(train_data)
test_data = add_padding_idx(test_data)




# Note that, no matter how many node types are here, make sure the
# hyperedge (N1,N2,N3,...) has id, N1 < N2 < N3...
train_dict = parallel_build_hash(train_data, "build_hash", args, num, initial = set())
test_dict = parallel_build_hash(test_data, "build_hash", args, num, initial = train_dict)
print ("dict_size", len(train_dict), len(test_dict))

# dict2 = build_hash2(train_data)
# pos_edges = list(dict2)
# pos_edges = np.array(pos_edges)
# np.random.shuffle(pos_edges)

print("train data amount", len(train_data))
# potential_outliers = build_hash3(np.concatenate((train_data, test), axis=0))
# potential_outliers = np.array(list(potential_outliers))


with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
        if args.feature == 'walk':
            # Note that for this part, the word2vec still takes sentences with
            # words starts at "0"
            if not args.TRY and os.path.exists(
                    "../%s_wv_%d_%s.npy" %
                    (args.data, args.dimensions, args.walk)):
                A = np.load(
                    "../%s_wv_%d_%s.npy" %
                    (args.data,
                     args.dimensions,
                     args.walk),
                    allow_pickle=True)
            else:
                print("start loading")
                walks = np.loadtxt(walk_path, delimiter=" ").astype('int')
                start = time.time()
                split_num = 20
                pool = ProcessPoolExecutor(max_workers=split_num)
                process_list = []
                walks = np.array_split(walks, split_num)
                
                result = []
                print("Start turning path to strs")
                for walk in walks:
                    process_list.append(pool.submit(walkpath2str, walk))
                
                for p in as_completed(process_list):
                    result += p.result()
                
                pool.shutdown(wait=True)
                
                walks = result
                print(
                    "Finishing Loading and processing %.2f s" %
                    (time.time() - start))
                print("Start Word2vec")
                import multiprocessing
                
                print("num cpu cores", multiprocessing.cpu_count())
                w2v = Word2Vec(
                    walks,
                    size=args.dimensions,
                    window=args.window_size,
                    min_count=0,
                    sg=1,
                    iter=1,
                    workers=multiprocessing.cpu_count())
                wv = w2v.wv
                A = [wv[str(i)] for i in range(num_list[-1])]
                np.save("../%s_wv_%d_%s.npy" %
                        (args.data, args.dimensions, args.walk), A)
                
                from sklearn.preprocessing import StandardScaler
                
                A = StandardScaler().fit_transform(A)
            
            A = np.concatenate(
                (np.zeros((1, A.shape[-1]), dtype='float32'), A), axis=0)
            A = A.astype('float32')
            A = torch.tensor(A).to(device)
            print(A.shape)
            
            node_embedding = Wrap_Embedding(int(
                num_list[-1] + 1), args.dimensions, scale_grad_by_freq=False, padding_idx=0, sparse=False)
            node_embedding.weight = nn.Parameter(A)
        
        elif args.feature == 'adj':
            flag = False
            
            node_embedding = MultipleEmbedding(
                embeddings_initial,
                bottle_neck,
                flag,
                num_list,
                node_type_mapping).to(device)
        
        classifier_model = Classifier(
            n_head=8,
            d_model=args.dimensions,
            d_k=16,
            d_v=16,
            node_embedding=node_embedding,
            diag_mask=args.diag,
            bottle_neck=bottle_neck).to(device)
        
        save_embeddings(classifier_model, True)
        
        Randomwalk_Word2vec = Word2vec_Skipgram(dict_size=int(num_list[-1] + 1), embedding_dim=args.dimensions,
                                                window_size=args.window_size, u_embedding=node_embedding,
                                                sparse=False).to(device)
        
        loss = F.binary_cross_entropy
        loss2 = torch.nn.BCEWithLogitsLoss(reduction='sum')
        
        summary(classifier_model, (3,))
    
        try:
            from datapipe import Word2Vec_Skipgram_Data
            sentences = Word2Vec_Skipgram_Data(train_data=walk_path,
                                               num_samples=neg_num_w2v,
                                               batch_size=128,
                                               window_size=args.window_size,
                                               min_count=0,
                                               subsample=1e-3,
                                               session=session)
        except:
            sentences = Word2Vec_Skipgram_Data_Empty()
        
        params_list = list(set(list(classifier_model.parameters()) + list(Randomwalk_Word2vec.parameters())))
        
        if args.feature == 'adj':
            optimizer = torch.optim.Adam(params_list, lr=1e-3)
        else:
            optimizer = torch.optim.RMSprop(params_list, lr=1e-3)
        
        model_parameters = filter(lambda p: p.requires_grad, params_list)
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("params to be trained", params)
        
        train(args, (classifier_model, Randomwalk_Word2vec),
              loss=((loss, 1.0), (loss2, 0.0)),
              training_data=(train_data, train_weight, sentences),
              validation_data=(test_data, test_weight),
              optimizer=[optimizer], epochs=300, batch_size=batch_size, only_rw=False)

