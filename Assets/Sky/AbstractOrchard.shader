
Shader "Skybox/Abstract Orchard"
{
    Properties
    {
        _MainTex1 ("tex2D", 2D) = "white" {}
        _MainTex2 ("tex2D", 2D) = "white" {}
        _MainTex3 ("tex2D", 2D) = "white" {}
        _SunDir ("Sun Dir", Vector) = (-.11,.07,0.99,0) 
        _XYZPos ("XYZ Offset", Vector) = (0, 15, -.25 ,0) 
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            uniform sampler2D _MainTex1,_MainTex2,_MainTex3; 
            float4 _SunDir,_XYZPos;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 worldPos : TEXCOORD1;
                float4 vertex : SV_POSITION;   //pos
                float4 screenPos : TEXCOORD2;
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                o.worldPos = mul(unity_ObjectToWorld, v2.vertex);
                o.screenPos  = ComputeScreenPos(o.vertex);
                //o.screenPos.z = -(mul(UNITY_MATRIX_MV, vertex).z * _ProjectionParams.w);     // old calculation, as I used the depth buffer comparision for min max ray march. 

                return o;
            }


// Abstract Orchard
// by Dave Hoskins. October 2019
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

int spointer;
float3 sunLight;
#define SUN_COLOUR float3(1., .9, .8)
#define FOG_COLOUR float3(1., .7, .7)

struct Stack
{
    float3 pos;
    float alpha;
    float dist;
    int mat;

};

#define STACK_SIZE 8
Stack stack[8];
Stack stack0;
Stack stack1;
Stack stack2;
Stack stack3;
Stack stack4;
Stack stack5;
Stack stack6;
Stack stack7;

float3 stack0pos;
float3 stack1pos;
float3 stack2pos;
float3 stack3pos;
float3 stack4pos;
float3 stack5pos;
float3 stack6pos;
float3 stack7pos;

float stack0alpha;
float stack1alpha;
float stack2alpha;
float stack3alpha;
float stack4alpha;
float stack5alpha;
float stack6alpha;
float stack7alpha;

float stack0dist;
float stack1dist;
float stack2dist;
float stack3dist;
float stack4dist;
float stack5dist;
float stack6dist;
float stack7dist;

int stack0mat;
int stack1mat;
int stack2mat;
int stack3mat;
int stack4mat;
int stack5mat;
int stack6mat;
int stack7mat;
//==============================================================================
//--------------------------------------------------------------------------
float getGroundHeight(float2 p)
{
    float y =(sin(p.y*.23)+cos(p.x*.18))*.8;
    return y;
}
//--------------------------------------------------------------------------
float3x3 getCamMat( in float3 ro, in float3 ta, float cr )
{
    float3 cw = normalize(ta-ro);
    float3 cp = float3(sin(cr), cos(cr),0.0);
    float3 cu = normalize( cross(cw,cp) );
    float3 cv = normalize( cross(cu,cw) );
    return float3x3( cu, cv, cw );
}

//--------------------------------------------------------------------------
// Loop the camposition around a uneven sine and cosine, and default the time 0
// to be steep at a loop point by adding 140...
float3 getCamPos(float t)
{
    //t = sin(t*.01)*200.;
    t+=140.;
    float3 p = float3(3.0+50.0*sin(t*.03),
                  1.5,
                  4.0 + 50.0*cos(t*.044));
    p.y-=getGroundHeight(p.xz);
    return p;
}

//----------------------------------------------------------------------------------------
//  1 out, 1 in...
float hash11(float p)
{
    p = frac(p * .1031);
    p *= p + 33.33;
    p *= p + p;
    return frac(p);
}
//----------------------------------------------------------------------------------------
//  1 out, 2 in...
float hash12(float2 p)
{
    float3 p3  = frac(float3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

//  1 out, 3 in...
float3 hash31(float p)
{
   float3 p3 = frac(float3(p,p,p) * float3(.1031, .1030, .0973));
   p3 += dot(p3, p3.yzx+33.33);
   return frac((p3.xxy+p3.yzz)*p3.zyx); 
}

//  3 out, 3 in...
float3 hash33(float3 p3)
{
    p3 = frac(p3 * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz+33.33);
    return frac((p3.xxy + p3.yxx)*p3.zyx);
}


//------------------------------------------------------------------------------
float randomTint(float3 pos)
{
    float r = tex2D(_MainTex1, pos.xz*.0027).x;
    return r+.5;
}

//----------------------------------------------------------------------------------------
float3 texCube(sampler2D sam, in float3 p, in float3 n )
{
    float3 x = tex2D(sam, p.yz).xyz;
    float3 y = tex2D(sam, p.zx).xyz;
    float3 z = tex2D(sam, p.xy).xyz;
    return (x*abs(n.x) + y*abs(n.y) + z*abs(n.z))/(abs(n.x)+abs(n.y)+abs(n.z));
}

//------------------------------------------------------------------------------
float4 grasstex2D(float3 pos, float3 nor)
{
    
    float g = tex2D(_MainTex1, pos.xz*.5).x;
    float s = tex2D(_MainTex1, pos.xz*.015).x*.2;
    
    
    float3 flower = tex2D(_MainTex2, pos.xz*.15).xyz;
    float rand = tex2D(_MainTex1, pos.xz*.003).x;
    rand *= rand*rand;
    
    flower =pow(flower,float3(8, 15, 5)) *10. * rand;
    float4 mat = float4(g*.05+s, g*.65, 0, g*.1);
    mat.xyz += flower;
    
    return min(mat, 1.0);
}

//------------------------------------------------------------------------------
float4 barktexture(float3 p, float3 nor)
{
    float2 r = floor(p.xz / 5.0) * 0.02;
    
    float br = tex2D(_MainTex1, r).x;
    float3 mat = texCube(_MainTex3, p, nor) * float3(.35, .25, .25);
    mat += texCube(_MainTex3, p*1.73, nor)*smoothstep(0.0,.2, mat.y)*br * float3(1,.9,.8);
    //mat*=mat*2.5;
    return float4(mat, .1);
}

//------------------------------------------------------------------------------
float4 leavestex2D(float3 p, float3 nor)
{
    
    float3 rand = texCube(_MainTex2, p*.15,nor);
    float3 mat = float3(0.4,1.2,0) *rand;
    if (nor.y < 0.0) mat += float3(1., 0.5,.5);
    
    return float4(mat, .0);
}

//------------------------------------------------------------------------------
float4 fruittex2D(float3 p, float3 nor, float i)
{
    
    
    float rand = texCube(_MainTex2, p*.1 ,nor).x;
    float t = dot(nor, normalize(float3(.8, .1, .1)));
    float3 mat = float3(1.,abs(t)*rand,0);
    mat = lerp(float3(0,1,0), mat, i/10.);

    return float4(mat, .5);
}



//------------------------------------------------------------------------------
float distanceRayPoint(float3 ro, float3 rd, float3 p, out float h)
{
    h = dot(p-ro,rd);
    return length(p-ro-rd*h);
}

//------------------------------------------------------------------------------
const int   SEEDS = 8 ;
const float STEP_SIZE = 2.;
#define SIZE .03


// This seed code is the starfield stuff from iapafoto
// I've just removed the alpha part...
// https://www.shadertoy.com/view/Xl2BRR
float2x2 rotfloat2x2D(float a)
{
    float si = sin(a);
    float co = cos(a);
    return float2x2(si, co, -co, si);
}

float3 floatingSeeds(in float3 ro, in float3 rd, in float tmax)
{ 
 
    float d =  0.;
    ro /= STEP_SIZE;
    float3 pos = floor(ro),
         ri = 1./rd,
         rs = sign(rd),
         dis = (pos-ro + .5 + rs*0.5) * ri;
    
    float dint;
    float3 offset, id;
    float3 col = float3(0,0,0);
    float3 sum = float3(0,0,0);
    //float size = .04;
    
    for( int i=0; i< SEEDS; i++ )
    {
        id = hash33(pos);

        offset = clamp(id+.2*cos(id*_Time.y),SIZE, 1.-SIZE);
        d = distanceRayPoint(ro, rd, pos+offset, dint);
        
        if (dint > 0. && dint * STEP_SIZE < tmax)
        {
            col = float3(.4,.4,.4)*smoothstep(SIZE, 0.0,d);
            sum += col;
        }
        float3 mm = step(dis.xyz, dis.yxy) * step(dis.xyz, dis.zzx);
        dis += mm * rs * ri;
        pos += mm * rs;
    }
  
    return sum * .7;
}

//--------------------------------------------------------------------------
float findClouds2D(in float2 p)
{
    float a = 1.5, r = 0.0;
    p*= .000001;
    for (int i = 0; i < 5; i++)
    {
        r+= tex2D(_MainTex1,p*=2.2).x*a;
        a*=.5;
    }
    return max(r-1.5, 0.0);
}
//------------------------------------------------------------------------------
// Use the difference between two cloud densities to light clouds in the direction of the sun.
float4 getClouds(float3 pos, float3 dir)
{
    if (dir.y < 0.0) return float4(0.0,0,0,0);
    float d = (4000. / dir.y);
    float2 p = pos.xz+dir.xz*d;
    float r = findClouds2D(p);
    float t = findClouds2D(p+normalize(sunLight.xz)*30.);    
    t = sqrt(max((r-t)*20., .2))*2.;
    float3 col = float3(t,t,t) * SUN_COLOUR;
    // returns colour and alpha...
    return float4(col, r);
} 

float mod(float a, float b) {return a%b;}

//------------------------------------------------------------------------------
// Thanks to Fizzer for the space-folded tree idea...
/// https://www.shadertoy.com/view/4tVcWR
float2 map(float3 p, float t)
{
 
    float matID, f;
    p.y += getGroundHeight(p.xz);
    float num = (floor(p.z/5.))*5.+(floor(p.x/5.0))*19.;
    p.xz = mod(p.xz, 5.0)-2.5;
    //p.xz *= rotfloat2x2D(p.y*num/300.); // ... No, just too expensive. :)
    
    float d = p.y;
    matID = 0.0;

    float s=1.,ss=1.6;
    
    // Tangent Vectors for the branch local coordinate system.
    float3 w=normalize(float3(-1.5+abs(hash11(num*4.)*.8),1,-1.));
    float3 u=normalize(cross(w,float3(0,1.,0.)));

    float scale=3.5;
    p/=scale;
    float3 q = p;
    // Make the iterations lessen over distance for speed up...
    int it = 10-int(min(t*.03, 9.0));

    float h  = hash11(num*7.)*.3+.3;
    float3 uwc = normalize(cross(u,w));
    int dontFold = int(hash11(num*23.0) * 9.0)+3;
    
    float thick = .2/(h-.24);
    for (int i = 0; i < it; i++)
    {
        f = scale*max(p.y-h,max(-p.y,length(p.xz)-.06/(p.y+thick)))/s;
        if (f <= d)
        {
            d = f;
            matID = 1.0;
        }

        // Randomly don't fold the space to give more branch types...
        if (i != dontFold)
            p.xz = abs(p.xz);

        p.y-=h;
        p=mul(p,float3x3(u,uwc,w));
        p*=ss;
        s*=ss;
    }

    float fr = .2;
    f = (length(p)-fr)/s;
    if (f <= d)
    {
        d = f;
        matID = 2.0;
    }
    
    q.y -= h*1.84;
    h *= 1.1;
    for (int i = 0; i < it; i++)
    {
        p = (normalize(hash31(num+float(i+19))-.5))*float3(h, 0.1, h);
        p+=q;
        float ds =length(p)-.015;
        if (ds <= d)
        {
            matID = 3.0+float(i);
            d = ds;
        }
    }

    return float2(d, matID);
}

//------------------------------------------------------------------------------
float sphereRadius(float t)
{
    t = abs(t-.0);
    t *= 0.003;
    return clamp(t, 1.0/450, 3000.0/450);  //return clamp(t, 1.0/iResolution.y, 3000.0/iResolution.y);
}

//------------------------------------------------------------------------------
float shadow( in float3 ro, in float3 rd, float dis)
{
    float res = 1.0;
    float t = hash11(dis)*.5+.2;
    float h;
    
    for (int i = 0; i < 10; i++)
    {
        float3 p =  ro + rd*t;

        h = map(p,dis).x;
        res = min(10.*h / t*t, res);
        t += h*2.5;
    }
    //res += t*t*.02; // Dim over distance
    return clamp(res, .3, 1.0);
}

//-------------------------------------------------------------------------------------------
// Taken almost straight from Inigo's shaders, thanks man!
// But I've changed a few things like the for-loop is now a float,
// which removes the need for the extra multiply and divide in GL2
float calcOcc( in float3 pos, in float3 nor, float d )
{
    float occ = 0.0;
    float sca = 1.0;

    for(float h= 0.07; h < .25; h+= .07)
    {
        float3 opos = pos + h*nor;
        d = map( opos, d ).x;
        occ += (h-d)*sca;
        sca *= 0.95;
    }
    return clamp( 1.0 - 2.0*occ, 0.0, 1.0 );
}

//-------------------------------------------------------------------------------------------
float marchScene(in float3 rO, in float3 rD, float2 co)
{
    float t = hash12(co)*.5;
    float4 normal = float4(0.0,0,0,0);
    float3 p;
    float alphaAcc = 0.0;

    spointer = 0;
    for( int j=0; j < 140; j++ )        //for( int j=0; j < 140; j++ )
    {
        // Check if it's full or too far...
        if (spointer == STACK_SIZE || alphaAcc >= 1.) break;
        p = rO + t*rD;
        float sphereR = sphereRadius(t);
        float2 h = map(p, t);
        if( h.x <= sphereR)
        {
            //h = max(h,0.0);
            float alpha = (1.0 - alphaAcc) * min(((sphereR-h.x+.01) / sphereR), 1.0);
            if (spointer == 0) {
                stack0pos = p;
                stack0alpha = alpha;
                stack0dist = t;
                stack0mat = int(h.y);
            }
            if (spointer == 1) {
                stack1pos = p;
                stack1alpha = alpha;
                stack1dist = t;
                stack1mat = int(h.y);
            }
            if (spointer == 2) {
                stack2pos = p;
                stack2alpha = alpha;
                stack2dist = t;
                stack2mat = int(h.y);
            }
            if (spointer == 3) {
                stack3pos = p;
                stack3alpha = alpha;
                stack3dist = t;
                stack3mat = int(h.y);
            }
            if (spointer == 4) {
                stack4pos = p;
                stack4alpha = alpha;
                stack4dist = t;
                stack4mat = int(h.y);
            }
            if (spointer == 5) {
                stack5pos = p;
                stack5alpha = alpha;
                stack5dist = t;
                stack5mat = int(h.y);
            }
            if (spointer == 6) {
                stack6pos = p;
                stack6alpha = alpha;
                stack6dist = t;
                stack6mat = int(h.y);
            }
            if (spointer == 7) {
                stack7pos = p;
                stack7alpha = alpha;
                stack7dist = t;
                stack7mat = int(h.y);
            }
            alphaAcc += alpha;
            spointer++;
        }
        t +=  h.x+t*0.007;
    }
    return alphaAcc;
}   

//-------------------------------------------------------------------------------------------
float3 lighting(in int id, in float4 mat, in float3 pos, in float3 normal, in float3 eyeDir, in float d)
{

    // Shadow and local occlusion...
    float sh = shadow(pos+sunLight*.01,  sunLight, d);
    float occ = calcOcc(pos, normal, d);
    
    // Light surface with 'sun'...
    float3 col = mat.xyz * SUN_COLOUR*(max(dot(sunLight,normal), 0.0))*sh;
    if (id == 2 && normal.y < -0.5)
    {
        col.y += .15;
    }


    // Ambient...
    
    float fre = clamp(1.0+dot(normal,eyeDir),0.0,1.0);
    
    float bac = clamp(dot( normal, normalize(float3(-sunLight.x,0.0,-sunLight.z))), 0.0, 1.0 );
    normal = reflect(eyeDir, normal); // Specular...
    col += pow(max(dot(sunLight, normal), 0.0), 16.0)  * SUN_COLOUR * sh * mat.w * occ * fre;
    col += bac*mat.xyz* .3 * occ;
    col += mat.xyz * max(normal.y, 0.0) * occ * .5;
    //col += SUN_COLOUR * fre *.2*occ;

    return min(col, 1.0);
}

//------------------------------------------------------------------------------
float3 getNormal2(float3 p, float e)
{
    return normalize( float3( map(p+float3(e,0.0,0.0), 0.).x - map(p-float3(e,0.0,0.0), 0.).x,
                            map(p+float3(0.0,e,0.0), 0.).x - map(p-float3(0.0,e,0.0), 0.).x,
                            map(p+float3(0.0,0.0,e), 0.).x - map(p-float3(0.0,0.0,e), 0.).x));
}

float3 getNormal(float3 pos, float ds)
{

    float c = map(pos, 0.).x;
    // Use offset samples to compute gradient / normal
    float2 eps_zero = float2(ds, 0.0);
    return normalize(float3(map(pos + eps_zero.xyy, 0.0).x, map(pos + eps_zero.yxy, 0.0).x,
                          map(pos + eps_zero.yyx, 0.0).x) - c);
}


//------------------------------------------------------------------------------
float3 getSky(float3 dir)
{
    float3 col = lerp(float3(FOG_COLOUR), float3(.0, 0.2,0.4),(abs(dir.y)));
    return col;
}



         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex1, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin

//    float2 mouseXY = iMouse.xy / iResolution.xy;
//    float2 uv = (-iResolution.xy + 2.0 * fragCoord ) / iResolution.y;
    sunLight = normalize(float3(_SunDir.xyz));            //-.8,1.8,-1.5));
    
    // Camera stuff...
    float time = _Time.y; //+mouseXY.x*100.0;
    float3 camera = ro; //getCamPos(time);
//    float3 lookat = getCamPos(time+10.);
//    float ride = sin(_Time.y*.3)+.3;
//    camera.y += ride;
//    lookat.y += ride;
    
//    float3x3 camMat = getCamMat(camera, lookat, 0.0);
//    float3 seedDir = normalize( float3(uv, cos((length(uv*.4)))));
//    float3 rd = camMat * seedDir;

    float3 col = float3(0,0,0);

    float3 sky  = getSky(rd);
  

    // Build the stack returning the final alpha value...
    float alpha = marchScene(camera, rd, fragCoord);
    float4 mat;
    // Render the stack...
    if (alpha > .0)
    {
        [unroll(8)]
        for (int i = 0; i < spointer; i++)
        {
            float3 pos = float3(0,0,0);
            float d=0;
            int matID;
            float alpha;

            //float3  pos = stack[i].pos; 
            //float d = stack[i].dist;
            //int matID = stack[i].mat;
            float3 nor =  getNormal(pos, sphereRadius(d));
            if (i==0) {
                pos = stack0pos; 
                d = stack0dist;
                matID = stack0mat;
                alpha = stack0alpha;
            }
            if (i==1) {
                pos = stack1pos; 
                d = stack1dist;
                matID = stack1mat;
                alpha = stack1alpha;
            }
            if (i==2) {
                pos = stack2pos; 
                d = stack2dist;
                matID = stack2mat;
                alpha = stack2alpha;
            }
            if (i==3) {
                pos = stack3pos; 
                d = stack3dist;
                matID = stack3mat;
                alpha = stack3alpha;
            }
            if (i==4) {
                pos = stack4pos; 
                d = stack4dist;
                matID = stack4mat;
                alpha = stack4alpha;
            }
            if (i==5) {
                pos = stack5pos; 
                d = stack5dist;
                matID = stack5mat;
                alpha = stack5alpha;
            }
            if (i==6) {
                pos = stack6pos; 
                d = stack6dist;
                matID = stack6mat;
                alpha = stack6alpha;
            }
            if (i==7) {
                pos = stack7pos; 
                d = stack7dist;
                matID = stack7mat;
                alpha = stack7alpha;
            }

            if (matID == 0) mat =  grasstex2D(pos, nor);
            else
                if (matID == 1) mat = barktexture(pos, nor);
            else
            if (matID == 2)
            {
                mat = leavestex2D(pos, nor);
                
                
            }
            else
                mat = fruittex2D(pos, nor, float(matID - 3));

            mat *= randomTint(pos);
 
            float3  temp = lighting(matID,mat, pos, nor, rd, d);
            if (matID == 3) temp=temp*.4+float3(.15, .01,0);
            
            temp = lerp(sky, temp , exp(-d*.01));
            col += temp *  alpha;    //stack[i].alpha;
        }
    }
    float4 cc = getClouds(camera, rd);
    sky+= pow(max(dot(sunLight, rd), 0.0), 300.0)*SUN_COLOUR*2.;
    sky = lerp(sky, cc.xyz, cc.w);
    col += sky *  (1.0-alpha);
    
    float d = stack0dist;
    col+= floatingSeeds(camera, rd, d);
    
   
    // Sun glow effect...
    col+=pow(max(dot(sunLight, rd), 0.0), 6.0)*SUN_COLOUR*.2;
    
    // Clamp and contrast...
    col = col * float3(1., .9,.9);
    col = clamp(col,0.,1.);
    col = col*col*(3.0-2.0*col);

    
    // The usual vignette...which manages to add more vibrancy...
//    float2 q = fragCoord / iResolution.xy;
//    col *= pow(90.0*q.x*q.y*(1.0-q.x)*(1.0-q.y), 0.5);
    // A nice fade in start...
    
    
    col *= smoothstep(0.0, 5.0, time);
    fragColor = float4(sqrt(col), 1.0);


                return fragColor;
            }

            ENDCG
        }
    }
}


