from torch import nn
from torch.autograd import Function
from utils import *
import torch
import bnmorph_getcorpts

torch.manual_seed(42)
class BNMorphFunction(Function):
    @staticmethod
    def find_corresponding_pts(binMapsrc, binMapdst, xx, yy, sxx, syy, cxx, cyy, pixel_distance_weight, alpha_distance_weight, pixel_mulline_distance_weight, alpha_padding):
        binMapsrc = binMapsrc.float()
        binMapdst = binMapdst.float()
        pixel_distance_weight = float(pixel_distance_weight)
        alpha_distance_weight = float(alpha_distance_weight)
        alpha_padding = float(alpha_padding)
        pixel_mulline_distance_weight = float(pixel_mulline_distance_weight)
        orgpts_x, orgpts_y, correspts_x, correspts_y, morphedx, morphedy = bnmorph_getcorpts.bnmorph(binMapsrc, binMapdst, xx, yy, sxx, syy, cxx, cyy, pixel_distance_weight, alpha_distance_weight, pixel_mulline_distance_weight, alpha_padding)
        ocoeff = dict()
        ocoeff['orgpts_x'] = orgpts_x
        ocoeff['orgpts_y'] = orgpts_y
        ocoeff['correspts_x'] = correspts_x
        ocoeff['correspts_y'] = correspts_y
        return morphedx, morphedy, ocoeff

class BNMorph(nn.Module):
    def __init__(self, height, width, serachWidth=7, searchHeight=3, sparsityRad=2, senseRange=20, pixel_distance_weight=24, alpha_distance_weight=0.7, pixel_mulline_distance_weight=1.9, alpha_padding=1.6):
        super(BNMorph, self).__init__()
        self.height = height
        self.width = width
        self.searchWidth = serachWidth
        self.searchHeight = searchHeight
        self.sparsityRad = sparsityRad
        self.senseRange = senseRange
        self.pixel_distance_weight = pixel_distance_weight
        self.alpha_distance_weight = alpha_distance_weight
        self.alpha_padding = alpha_padding
        self.pixel_mulline_distance_weight = pixel_mulline_distance_weight

        self.pixel_distance_weight_store = None
        self.alpha_distance_weight_store = None
        self.pixel_mulline_distance_weight_store = None
        self.alpha_padding_store = None

        colsearchSpan = np.arange(-self.searchHeight, self.searchHeight + 1)
        rowsearchSpan = np.arange(-self.searchWidth, self.searchWidth + 1)
        xx, yy = np.meshgrid(rowsearchSpan, colsearchSpan)
        xx = xx.flatten()
        yy = yy.flatten()
        dist = xx**2 + yy**2
        sortedInd = np.argsort(dist)
        self.xx = torch.nn.Parameter(torch.from_numpy(xx[sortedInd]).float(), requires_grad=False)
        self.yy = torch.nn.Parameter(torch.from_numpy(yy[sortedInd]).float(), requires_grad=False)

        sparsittSpan = np.arange(-self.sparsityRad, self.sparsityRad + 1)
        sxx, syy = np.meshgrid(sparsittSpan, sparsittSpan)
        self.sxx = torch.nn.Parameter(torch.from_numpy(sxx.flatten()).float(), requires_grad=False)
        self.syy = torch.nn.Parameter(torch.from_numpy(syy.flatten()).float(), requires_grad=False)

        senseSpan = np.arange(-self.senseRange, self.senseRange + 1)
        cxx, cyy = np.meshgrid(senseSpan, senseSpan)
        cxx = cxx.flatten()
        cyy = cyy.flatten()
        dist = cxx ** 2 + cyy ** 2
        sortedInd = np.argsort(dist)
        self.cxx = torch.nn.Parameter(torch.from_numpy(cxx[sortedInd]).float(), requires_grad=False)
        self.cyy = torch.nn.Parameter(torch.from_numpy(cyy[sortedInd]).float(), requires_grad=False)

    def find_corresponding_pts(self, binMapsrc, binMapdst):
        return BNMorphFunction.find_corresponding_pts(binMapsrc, binMapdst, self.xx, self.yy, self.sxx, self.syy, self.cxx, self.cyy, self.pixel_distance_weight, self.alpha_distance_weight, self.pixel_mulline_distance_weight, self.alpha_padding)