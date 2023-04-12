from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy import units as u
from astroquery.utils import async_to_sync
ra=151.8650810230822
dec=7.144407135105523
pos = coords.SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
xid = SDSS.query_region(pos, spectro=True)
# print(xid)
# im = async_to_sync(SDSS.get_images)(matches=xid, band='grz')
# for i in range(len(im)):
#     im[i].writeto('/data/public/renhaoye/image_%d.fits' % i, overwrite=True)
# 查看xid中的所有信息
print(xid.colnames)
# print(xid)
# print(xid['ra', 'dec', 'z', 'plate', 'mjd', 'fiberID'])
# print(xid['ra', 'dec', 'z', 'plate', 'mjd', 'fiberID'][0])
# print(xid['ra', 'dec', 'z', 'plate', 'mjd', 'fiberID'][0][0])
